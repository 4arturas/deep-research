import 'dotenv/config';
import { z } from 'zod';
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  ToolMessage,
  filterMessages
} from '@langchain/core/messages';
import { ChatOllama } from '@langchain/ollama';
import { StateGraph, START, END, Command, Annotation } from '@langchain/langgraph';
import { tool } from '@langchain/core/tools';
import { TavilySearch } from '@langchain/tavily';

import {
  research_agent_prompt,
  compress_research_system_prompt,
  compress_research_human_message,
  summarize_webpage_prompt
} from './prompts.js';

/* =============================================================================
   1. UTILS & STATE
   ========================================================================== */

export const getTodayStr = () => new Date().toLocaleDateString('en-US', {
  weekday: 'short', month: 'short', day: 'numeric', year: 'numeric'
});

export const SummarySchema = z.object({
  summary: z.string().describe('Concise summary of the webpage content'),
  key_excerpts: z.string().describe('Important quotes and excerpts from the content'),
});

export const ResearcherStateAnnotation = Annotation.Root({
  researcher_messages: Annotation({ reduce: (prev, next) => prev.concat(next), default: () => [] }),
  research_topic: Annotation({ reduce: (_, next) => next, default: () => '' }),
  compressed_research: Annotation({ reduce: (_, next) => next, default: () => '' }),
  raw_notes: Annotation({ reduce: (prev, next) => prev.concat(next), default: () => [] }),
  tool_call_iterations: Annotation({ reduce: (_, next) => next, default: () => 0 }),
});

/* =============================================================================
   2. TOOLS
   ========================================================================== */

const summarizationModel = new ChatOllama({
  model: process.env.OLLAMA_MODEL || 'llama3:8b',
  temperature: 0,
});

async function summarizeWebpageContent(webpageContent) {
  try {
    const structuredLLM = summarizationModel.withStructuredOutput(SummarySchema);
    const prompt = summarize_webpage_prompt
        .replace('{webpage_content}', webpageContent)
        .replace('{date}', getTodayStr());

    const result = await structuredLLM.invoke([new HumanMessage(prompt)]);
    return `<summary>\n${result.summary}\n</summary>\n\n<key_excerpts>\n${result.key_excerpts}\n</key_excerpts>`;
  } catch (e) {
    return webpageContent.slice(0, 1500) + "... [Truncated]";
  }
}

export const tavily_search = tool(
    async ({ query, maxResults = 3 }) => {
      const tvly = new TavilySearch({ apiKey: process.env.TAVILY_API_KEY, maxResults, includeRawContent: true });
      const response = await tvly.invoke(query);
      const data = typeof response === 'string' ? JSON.parse(response) : response;
      const results = data.results || [];

      if (results.length === 0) return "No results found.";

      let formattedOutput = `Search results for: "${query}"\n\n`;
      for (const [index, result] of results.entries()) {
        const processedContent = await summarizeWebpageContent(result.raw_content || result.content);
        formattedOutput += `--- SOURCE ${index + 1}: ${result.title} ---\nURL: ${result.url}\nCONTENT:\n${processedContent}\n---\n\n`;
      }
      return formattedOutput;
    },
    {
      name: 'tavily_search',
      description: 'Search the web for information and return summarized content.',
      schema: z.object({
        query: z.string(),
        maxResults: z.number().int().default(3),
      }),
    }
);

export const think_tool = tool(
    async ({ reflection }) => `Reflection recorded: ${reflection}`,
    {
      name: 'think_tool',
      description: 'Analyze search results and plan next steps.',
      schema: z.object({ reflection: z.string() }),
    }
);

const tools = [tavily_search, think_tool];
const toolsByName = Object.fromEntries(tools.map((t) => [t.name, t]));

/* =============================================================================
   3. NODES & GRAPH
   ========================================================================== */

const model = new ChatOllama({ model: 'llama3:8b', temperature: 0 }).bindTools(tools);
const compressModel = new ChatOllama({ model: 'llama3:8b', temperature: 0 });

async function llm_call(state) {
  const response = await model.invoke([
    new SystemMessage(research_agent_prompt.replace('{date}', getTodayStr())),
    ...state.researcher_messages,
  ]);

  const hasToolCalls = response.tool_calls?.length > 0;
  return new Command({
    goto: hasToolCalls ? 'tool_node' : 'compress_research',
    update: { researcher_messages: [response] },
  });
}

async function tool_node(state) {
  const lastMessage = state.researcher_messages[state.researcher_messages.length - 1];
  const toolOutputs = await Promise.all(
      (lastMessage.tool_calls || []).map(async (call) => {
        const obs = await toolsByName[call.name].invoke(call.args);
        return new ToolMessage({
          content: String(obs),
          name: call.name,
          tool_call_id: call.id
        });
      })
  );
  return new Command({ goto: 'llm_call', update: { researcher_messages: toolOutputs } });
}

async function compress_research(state) {
  const response = await compressModel.invoke([
    new SystemMessage(compress_research_system_prompt.replace('{date}', getTodayStr())),
    ...state.researcher_messages,
    new HumanMessage(compress_research_human_message.replace('{topic}', state.research_topic)),
  ]);

  const rawNotesList = filterMessages(state.researcher_messages, { includeTypes: ['tool', 'ai'] })
      .map((m) => String(m.content));

  return new Command({
    goto: END,
    update: {
      compressed_research: String(response.content),
      raw_notes: [rawNotesList.join('\n\n--- NEXT NOTE ---\n\n')],
    },
  });
}

export function buildResearchAgent() {
  return new StateGraph(ResearcherStateAnnotation)
      .addNode('llm_call', llm_call)
      .addNode('tool_node', tool_node)
      .addNode('compress_research', compress_research)
      .addEdge(START, 'llm_call')
      .compile();
}

export async function runResearchAgent(topic) {
  const app = buildResearchAgent();
  const eventStream = await app.stream({
    researcher_messages: [new HumanMessage(topic)],
    research_topic: topic,
  });

  let finalState = {};
  for await (const event of eventStream) {
    const nodeName = Object.keys(event)[0];
    finalState = { ...finalState, ...event[nodeName] };
  }
  return finalState;
}