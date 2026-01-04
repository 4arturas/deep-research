import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { ChatOllama } from '@langchain/ollama';
import { HumanMessage, ToolMessage, SystemMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { TavilySearch } from '@langchain/tavily';
import { MODEL_AGENT, MODEL_GENERATION } from './constants.js';

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

const researchAgentPrompt = `You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

<Task>
Your job is to use tools to gather information about the user's input topic.
You can use any of the tools provided to you to find resources that can help answer the research question. You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
</Task>

<Available Tools>
You have access to two main tools:
1. **tavily_search**: For conducting web searches to gather information
2. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool after each search to reflect on results and plan next steps**
</Available Tools>

<Instructions>
Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Start with broader searches** - Use broad, comprehensive queries first
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 2-3 search tool calls maximum
- **Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>`;

const compressResearchSystemPrompt = `You are a research assistant that has conducted research on a topic by calling several tools and web searches. Your job is now to clean up the findings, but preserve all of the relevant statements and information that the researcher has gathered. For context, today's date is {date}.

<Task>
You need to clean up information gathered from tool calls and web searches in the existing messages.
All relevant information should be repeated and rewritten verbatim, but in a cleaner format.
The purpose of this step is just to remove any obviously irrelevant or duplicate information.
For example, if three sources all say "X", you could say "These three sources all stated X".
Only these fully comprehensive cleaned findings are going to be returned to the user, so it's crucial that you don't lose any information from the raw messages.
</Task>

<Tool Call Filtering>
**IMPORTANT**: When processing the research messages, focus only on substantive research content:
- **Include**: All tavily_search results and findings from web searches
- **Exclude**: think_tool calls and responses - these are internal agent reflections for decision-making and should not be included in the final research report
- **Focus on**: Actual information gathered from external sources, not the agent's internal reasoning process

The think_tool calls contain strategic reflections and decision-making notes that are internal to the research process but do not contain factual information that should be preserved in the final report.
</Tool Call Filtering>

<Guidelines>
1. Your output findings should be fully comprehensive and include ALL of the information and sources that the researcher has gathered from tool calls and web searches. It is expected that you repeat key information verbatim.
2. This report can be as long as necessary to return ALL of the information that the researcher has gathered.
3. In your report, you should return inline citations for each source that the researcher found.
4. You should include a "Sources" section at the end of the report that lists all of the sources the researcher found with corresponding citations, cited against statements in the report.
5. Make sure to include ALL sources that the researcher gathered in the report, and how they were used to answer the question!
</Guidelines>

<Output Format>
The report should be structured like this:
**List of Queries and Tool Calls Made**
**Fully Comprehensive Findings**
**List of All Relevant Sources (with citations in the report)**
</Output Format>

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
</Citation Rules>

Critical Reminder: It is extremely important that any information that is even remotely relevant to the user's research topic is preserved verbatim (e.g. don't rewrite it, don't summarize it, don't paraphrase it).
`;

const compressResearchHumanMessage = `All above messages are about research conducted by an AI Researcher for the following research topic:

RESEARCH TOPIC: {research_topic}

Your task is to clean up these research findings while preserving ALL information that is relevant to answering this specific research question.

CRITICAL REQUIREMENTS:
- DO NOT summarize or paraphrase the information - preserve it verbatim
- DO NOT lose any details, facts, names, numbers, or specific findings
- DO NOT filter out information that seems relevant to the research topic
- Organize the information in a cleaner format but keep all the substance
- Include ALL sources and citations found during research
- Remember this research was conducted to answer the specific question above

The cleaned findings will be used for final report generation, so comprehensiveness is critical.`;

function createTavilySearch() {
  return new TavilySearch({
    maxResults: 3,
    includeRawContent: true,
    apiKey: process.env.TAVILY_API_KEY
  });
}

const thinkTool = tool(
  async ({ reflection }) => {
    console.log(`Reflection recorded: ${reflection}`);
    return `Reflection recorded: ${reflection}`;
  },
  {
    name: "think_tool",
    description: "Tool for strategic reflection on research progress and decision-making. Use this tool after each search to analyze results and plan next steps systematically.",
    schema: z.object({
      reflection: z.string().describe("Your detailed reflection on research progress, findings, gaps, and next steps")
    })
  }
);

const model = new ChatOllama({
  model: MODEL_AGENT,
  temperature: 0.0
});

function getTodayStr() {
  const now = new Date();
  return now.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
}

async function llmCall(state) {
  const systemPrompt = researchAgentPrompt.replace('{date}', getTodayStr());
  const messages = [new SystemMessage(systemPrompt), ...state.researcher_messages];
  
  let allTools = [thinkTool];
  if (process.env.TAVILY_API_KEY) {
    const tavilySearchTool = createTavilySearch();
    allTools = [tavilySearchTool, thinkTool];
  }
  
  const response = await model.bindTools(allTools).invoke(messages);
  
  return {
    researcher_messages: [response]
  };
}

async function toolNode(state) {
  const lastMessage = state.researcher_messages[state.researcher_messages.length - 1];
  const toolCalls = lastMessage.tool_calls || [];
  
  const observations = [];
  for (const toolCall of toolCalls) {
    let tool = null;
    
    if (toolCall.name === "tavily_search" && process.env.TAVILY_API_KEY) {
      tool = createTavilySearch();
    } else if (toolCall.name === "think_tool") {
      tool = thinkTool;
    }
    
    if (tool) {
      const result = await tool.invoke(toolCall.args);
      observations.push(result);
    } else {
      console.warn(`Tool ${toolCall.name} not found`);
      observations.push(`Error: Tool ${toolCall.name} not found`);
    }
  }
  
  const toolOutputs = observations.map((observation, index) => {
    const toolCall = toolCalls[index];
    return new ToolMessage({
      content: observation,
      name: toolCall.name,
      tool_call_id: toolCall.id
    });
  });
  
  return { researcher_messages: toolOutputs };
}

async function compressResearch(state) {
  const systemMessage = compressResearchSystemPrompt.replace('{date}', getTodayStr());
  const humanMessage = compressResearchHumanMessage.replace('{research_topic}', state.research_topic);
  
  const messages = [
    new SystemMessage(systemMessage),
    ...state.researcher_messages,
    new HumanMessage(humanMessage)
  ];
  
  const response = await model.invoke(messages);
  
  const rawNotes = state.researcher_messages
    .filter(msg => msg._getType && (msg._getType() === 'tool' || msg._getType() === 'ai'))
    .map(msg => msg.content.toString());
  
  return {
    compressed_research: response.content,
    raw_notes: [rawNotes.join('\n')]
  };
}

function shouldContinue(state) {
  const messages = state.researcher_messages;
  const lastMessage = messages[messages.length - 1];
  
  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    return "tool_node";
  }
  
  return "compress_research";
}

const agentBuilder = new StateGraph(ResearcherStateAnnotation);

agentBuilder.addNode("llm_call", llmCall);
agentBuilder.addNode("tool_node", toolNode);
agentBuilder.addNode("compress_research", compressResearch);

agentBuilder.addEdge(START, "llm_call");
agentBuilder.addConditionalEdges(
  "llm_call",
  shouldContinue,
  {
    "tool_node": "tool_node",
    "compress_research": "compress_research"
  }
);
agentBuilder.addEdge("tool_node", "llm_call");
agentBuilder.addEdge("compress_research", END);

const researcherAgent = agentBuilder.compile();

async function runResearchExample() {
  console.log("Starting research example...");
  
  const config = { configurable: { thread_id: "research-example" } };
  
  const input = {
    researcher_messages: [
      new HumanMessage("I want to research the best coffee shops in San Francisco based on coffee quality.")
    ],
    research_topic: "Best coffee shops in San Francisco based on coffee quality"
  };
  
  const result = await researcherAgent.invoke(input, config);
  
  console.log("Research completed!");
  console.log("Compressed Research:", result.compressed_research);
  console.log("Raw Notes:", result.raw_notes);
}

// runResearchExample().catch(console.error);

function evaluateNextStep(outputs, referenceOutputs) {
  const lastMessage = outputs.researcher_messages[outputs.researcher_messages.length - 1];
  const madeToolCall = lastMessage.tool_calls && lastMessage.tool_calls.length > 0;
  const shouldContinue = referenceOutputs.next_step === "continue";
  
  return {
    key: "correct_next_step",
    score: madeToolCall === shouldContinue
  };
}

async function targetFunc(inputs) {
  const config = { configurable: { thread_id: Math.random().toString() } };
  const result = await llmCall(inputs);
  return result;
}

export { researcherAgent, evaluateNextStep, targetFunc };