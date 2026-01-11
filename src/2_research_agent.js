import {StateGraph, START, END, Annotation, Command} from '@langchain/langgraph';
import { ChatOllama } from '@langchain/ollama';
import {HumanMessage, ToolMessage, SystemMessage, filterMessages, AIMessage} from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { TavilySearch } from '@langchain/tavily';
import {MODEL_AGENT, MODEL_GENERATION, MODEL_MINISTRAL_3_14B_CLOUD} from './constants.js';
import {
  compress_research_human_message, compress_research_system_prompt, research_agent_prompt
} from "./prompts.js";
import dayjs from "dayjs";
import fs from "fs";
import path from "path";

const ResearcherStateAnnotation = Annotation.Root({
  researcher_messages: Annotation({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  tool_call_iterations: Annotation({
    reducer: (x, y) => y,
    default: () => 0,
  }),
  research_topic: Annotation({
    reducer: (x, y) => y,
    default: () => "",
  }),
  compressed_research: Annotation({
    reducer: (x, y) => y,
    default: () => "",
  }),
  raw_notes: Annotation({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

async function invokeWithRetry(model, messages, retries = 5) {
  for (let i = 0; i < retries; i++) {
    try {
      console.log(`Attempting LLM call (Attempt ${i + 1})...`);
      return await model.invoke(messages);
    } catch (err) {
      if (i === retries - 1) throw err;
      const delay = Math.pow(2, i) * 1000;
      console.warn(`LLM failed with 500. Retrying in ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

// Drastically reduce content to save local memory
function truncateContent(content, maxChars = 800) {
  if (content.length <= maxChars) return content;
  return content.substring(0, maxChars) + "\n\n[Truncated...]";
}

const tavily_search = tool(
    async ({ query }) => {
      console.log(`Search query received: ${query}`);
      try {
        const filePath = 'C:\\Users\\4artu\\IdeaProjects\\deep-research\\src\\files\\coffee_shops_sf.md';
        if (fs.existsSync(filePath)) {
          const rawContent = fs.readFileSync(filePath, 'utf8');
          return truncateContent(rawContent);
        }
        return "No local data found.";
      } catch (err) {
        return `Error: ${err.message}`;
      }
    },
    {
      name: "tavily_search",
      description: "Search for coffee shops and quality.",
      schema: z.object({
        query: z.string().describe("The search query")
      })
    }
);

const think_tool = tool(
    async ({ reflection }) => {
      console.log(`Reflection: ${reflection}`);
      return `Plan updated.`;
    },
    {
      name: "think_tool",
      description: "Strategic planning tool.",
      schema: z.object({
        reflection: z.string().describe("Reflection content")
      })
    }
);

const getTodayStr = () => dayjs().format('ddd MMM DD, YYYY');

let model;
let compressModel;
let toolsByName;

async function initializeTools() {
  if (model) return;
  const allTools = [tavily_search, think_tool];
  toolsByName = Object.fromEntries(allTools.map((t) => [t.name, t]));
  model = new ChatOllama({ model: MODEL_AGENT, temperature: 0 }).bindTools(allTools);
  compressModel = new ChatOllama({ model: MODEL_MINISTRAL_3_14B_CLOUD, temperature: 0 });
}

async function llm_call(state) {
  await initializeTools();

  // CONTEXT MANAGEMENT:
  // We filter out old tool messages to prevent the 500 error.
  // We keep only the most recent pair of tool interactions and the current plan.
  const allMessages = state.researcher_messages;
  let contextMessages = [];

  if (allMessages.length > 6) {
    contextMessages = [
      allMessages[0], // Keep original instruction
      new AIMessage("I have performed several searches and refined my plan. I am continuing the research now."),
      ...allMessages.slice(-4) // Keep only last 2 rounds of interaction
    ];
  } else {
    contextMessages = allMessages;
  }

  const result = await invokeWithRetry(model, [
    new SystemMessage(research_agent_prompt.replace("{date}", getTodayStr())),
    ...contextMessages,
  ]);

  const hasToolCalls = result.tool_calls && result.tool_calls.length > 0;
  return new Command({
    goto: hasToolCalls ? "tool_node" : "compress_research",
    update: { researcher_messages: [result] },
  });
}

async function tool_node(state) {
  const lastMessage = state.researcher_messages[state.researcher_messages.length - 1];
  const toolCalls = lastMessage.tool_calls || [];

  const toolOutputs = await Promise.all(
      toolCalls.map(async (call) => {
        const tool = toolsByName[call.name];
        console.log(`Executing: ${call.name}`);
        let content;
        try {
          const obs = await tool.invoke(call.args);
          content = String(obs);
        } catch (err) {
          content = `Error: ${err.message}`;
        }
        return new ToolMessage({
          name: call.name,
          tool_call_id: call.id,
          content: content
        });
      })
  );

  return new Command({
    goto: "llm_call",
    update: { researcher_messages: toolOutputs }
  });
}

async function compress_research(state) {
  await initializeTools();
  const toolResults = state.researcher_messages
      .filter(m => m instanceof ToolMessage)
      .map(m => m.content)
      .join("\n\n");

  const response = await invokeWithRetry(compressModel, [
    new SystemMessage(compress_research_system_prompt.replace("{date}", getTodayStr())),
    new HumanMessage(`Topic: ${state.research_topic}\n\nFindings:\n${truncateContent(toolResults, 3000)}`)
  ]);

  return new Command({
    goto: END,
    update: {
      compressed_research: String(response.content),
      raw_notes: [toolResults],
    },
  });
}

async function runResearchExample() {
  console.log("Starting research example...");
  const builder = new StateGraph(ResearcherStateAnnotation)
      .addNode("llm_call", llm_call, { ends: ["tool_node", "compress_research"] })
      .addNode("tool_node", tool_node, { ends: ["llm_call"] })
      .addNode("compress_research", compress_research, { ends: [END] })
      .addEdge(START, "llm_call");

  const researcherAgent = builder.compile();
  const result = await researcherAgent.invoke({
    researcher_messages: [new HumanMessage("Research best coffee shops in SF by quality.")],
    research_topic: "Best coffee shops in SF"
  });

  console.log("Compressed Research:", result.compressed_research);
}

runResearchExample().catch(console.error);