import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { ChatOllama } from "@langchain/ollama";
import { HumanMessage, SystemMessage, ToolMessage, AIMessage, filterMessages } from "@langchain/core/messages";
import { StateGraph, START, END, Annotation, Command } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import path from "path";
import {MODEL_AGENT, MODEL_MINISTRAL_3_14B_CLOUD} from "./constants.js";
import {
    research_agent_prompt_with_mcp,
    compress_research_system_prompt,
    compress_research_human_message,
} from "./prompts.js";

const ResearcherStateAnnotation = Annotation.Root({
    researcher_messages: Annotation({
        reduce: (prev, next) => prev.concat(next),
        default: () => [],
    }),
    research_topic: Annotation({
        reduce: (_, next) => next,
        default: () => "",
    }),
    compressed_research: Annotation({
        reduce: (_, next) => next,
        default: () => "",
    }),
    raw_notes: Annotation({
        reduce: (prev, next) => prev.concat(next),
        default: () => [],
    }),
});

const think_tool = tool(
    async ({ reflection }) => `Reflection recorded: ${reflection}`,
    {
        name: "think_tool",
        description: "Analyze findings and plan next steps.",
        schema: z.object({ reflection: z.string() }),
    }
);

let model;
let compressModel;
let toolsByName;
let mcpClient;

async function initializeTools() {
    if (model)
        return;

    console.log("Initializing MCP Client...");
    mcpClient = new MultiServerMCPClient({
        filesystem: {
            transport: "stdio",
            command: "npx",
            args: [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                path.resolve(process.cwd(), "files"),
            ],
        },
    });

    const mcpTools = await mcpClient.getTools();
    const allTools = [...mcpTools, think_tool];
    toolsByName = Object.fromEntries(allTools.map((t) => [t.name, t]));

    model = new ChatOllama({ model: MODEL_AGENT, temperature: 0 }).bindTools(
        allTools
    );
    compressModel = new ChatOllama({ model: MODEL_MINISTRAL_3_14B_CLOUD, temperature: 0 });

    console.log(
        "Model initialized with tools:",
        // allTools.map((t) => t.name)
    );
    /*
    allTools.forEach(t => {
        console.log(`Tool ${t.name} schema:`, JSON.stringify(t.schema, null, 2));
    });
    */
}

function getTodayStr() {
    return new Date().toISOString().split("T")[0];
}

async function llm_call(state) {
    await initializeTools();
    const date = getTodayStr();
    const prompt = research_agent_prompt_with_mcp.replace("{date}", date);
    const result = await model.invoke([
        new SystemMessage(prompt),
        ...state.researcher_messages,
    ]);

    const hasToolCalls = result.tool_calls?.length > 0;

    return new Command({
        goto: hasToolCalls ? "tool_node" : "compress_research",
        update: { researcher_messages: [result] },
    });
}

async function tool_node( state )
{
    const messages = state.researcher_messages;
    const lastMessage = messages[messages.length-1];
    const toolCalls = lastMessage.tool_calls || [];

    const toolOutputs = await Promise.all(
        toolCalls.map( async call => {
            const tool = toolsByName[call.name];
            let content;
            try {
                const obs = await tool.invoke(call.args);
                content = String(obs);
            } catch (err) {
                content = `Error: ${err.message}`
            }
            return new ToolMessage({
                name: call.name,
                tool_call_id: call.id,
                content: content
            })
        })
    );

    return new Command({
        goto: "llm_call",
        update: {
            researcher_messages: toolOutputs
        }
    });
}

async function compress_research(state) {
    await initializeTools();
    const systemPrompt = compress_research_system_prompt.replace("{date}", getTodayStr());
    const humanPrompt = compress_research_human_message.replace("{research_topic}", state.research_topic);
    const response = await compressModel.invoke([
        new SystemMessage(systemPrompt),
        ...state.researcher_messages,
        new HumanMessage(humanPrompt),
    ]);

    const rawNotesList = filterMessages(state.researcher_messages, {
        includeTypes: ["tool", "ai"],
    })
        .map((m) => String(m.content))
        .join("\n\n--- NEXT NOTE ---\n\n");

    return new Command({
        goto: END,
        update: {
            compressed_research: String(response.content),
            raw_notes: [rawNotesList],
        },
    });
}

export async function runResearchAgentMCP(topic)
{
    const builder = new StateGraph(ResearcherStateAnnotation)
        .addNode("llm_call", llm_call, {ends: ["tool_node", "compress_research"]})
        .addNode("tool_node", tool_node)
        .addNode("compress_research", compress_research)
        .addEdge(START, "llm_call")
        .addEdge("tool_node", "llm_call");

    const app = builder.compile();
    const eventStream = await app.stream(
        {
            researcher_messages: [new HumanMessage(topic)],
            research_topic: topic,
        },
        {
            recursionLimit: 50,
        }
    );

    let finalState = {};
    for await (const event of eventStream) {
        const nodeName = Object.keys(event)[0];
        if (nodeName) {
            console.log(`Processing event from node: ${nodeName}`);
            finalState = { ...finalState, ...event[nodeName] };
        }
    }

    return finalState;
}

const topic = `
  I want to identify and evaluate the coffee shops in San Francisco that are considered the best based
  specifically
  on coffee quality. My research should focus on analyzing and comparing coffee shops within the San Francisco
  area,
  using coffee quality as the primary criterion. I am open regarding methods of assessing coffee quality (e.g.,
  expert reviews, customer ratings, specialty coffee certifications), and there are no constraints on ambiance,
  location, wifi, or food options unless they directly impact perceived coffee quality. Please prioritize primary
  sources such as the official websites of coffee shops, reputable third-party coffee review organizations (like
  Coffee Review or Specialty Coffee Association), and prominent review aggregators like Google or Yelp where
  direct
  customer feedback about coffee quality can be found. The study should result in a well-supported list or
  ranking of
  the top coffee shops in San Francisco, emphasizing their coffee quality according to the latest available data
  as
  of July 2025.
  `;
console.log(`Starting research agent with topic: "${topic.substring(0, 100)}..."`);
runResearchAgentMCP(topic)
    .then((result) => {
        console.log("\n\n=== Final Research Output ===\n");
        console.log(result.compressed_research);
        process.exit(0);
    });

