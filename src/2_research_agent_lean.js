import { HumanMessage, ToolMessage, AIMessage, getBufferString } from '@langchain/core/messages';
import { ChatOllama } from '@langchain/ollama';
import dayjs from 'dayjs';
import { z } from 'zod';
import { MODEL_QWEN } from "./constants.js";
import { research_agent_prompt, compress_research_system_prompt, compress_research_human_message } from "./prompts.js";
import { tool } from '@langchain/core/tools';
import * as fs from 'fs';

const getTodayStr = () => dayjs().format('ddd MMM DD, YYYY');

const tavily_search = tool(
    async ({ query }) => {
        console.log(`Search query received: ${query}`);
        // Mocking search with local file for "San Francisco coffee shops"
        // In a real scenario, this would call the Tavily API
        const filePath = 'C:\\Users\\4artu\\IdeaProjects\\deep-research\\src\\files\\coffee_shops_sf.md';

        try {
            const rawContent = fs.readFileSync(filePath, 'utf8');
            const maxChars = 800;
            if (rawContent.length <= maxChars) return rawContent;
            return rawContent.substring(0, maxChars) + "\n\n[Truncated...]";
        } catch (e) {
            console.error("Error reading file:", e);
            return "Error reading search results file.";
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

// Mock think_tool for strategic reflection
const think_tool = tool(
    async ({ reflection }) => {
        console.log(`Reflection: ${reflection}`);
        return `Reflection recorded: ${reflection}`;
    },
    {
        name: "think_tool",
        description: "Strategic planning tool.",
        schema: z.object({
            reflection: z.string().describe("Reflection content")
        })
    }
);

const model = new ChatOllama({ model: MODEL_QWEN, temperature: 0.0 });
const tools = [tavily_search, think_tool];
const toolsByName = Object.fromEntries(tools.map((t) => [t.name, t]));
const modelWithTools = model.bindTools(tools);
const compressModel = new ChatOllama({ model: MODEL_QWEN, temperature: 0.0 });

async function run() {
    const researchTopic = "I want to research coffee shops in San Francisco.";
    console.log(`🔍 Search query: "${researchTopic}"\n`);

    const messages = [new HumanMessage(researchTopic)];
    let toolCallIterations = 0;
    const maxIterations = 5;

    while (toolCallIterations < maxIterations) {
        const date = getTodayStr();
        const systemPrompt = research_agent_prompt.replace("{date}", date);
        const response = await modelWithTools.invoke([
            new HumanMessage(systemPrompt),
            ...messages
        ]);

        messages.push(response);

        const toolCalls = response.tool_calls || [];

        if (toolCalls.length > 0) {
            toolCallIterations++;

            for (const toolCall of toolCalls) {
                const tool = toolsByName[toolCall.name];

                console.log(`\n\n🔧 ======== Executing: ${toolCall.name} ========`);
                console.log(`Args: ${JSON.stringify(toolCall.args)}`);

                let result;
                try {
                    result = await tool.invoke(toolCall.args);
                } catch (error) {
                    console.error(`Error executing tool ${toolCall.name}:`, error);
                    result = `Error executing tool: ${error.message}`;
                }

                console.log(`Result: ${result.substring(0, 100)}...`);

                messages.push(new ToolMessage({
                    tool_call_id: toolCall.id,
                    content: result,
                    name: toolCall.name
                }));
            }
        } else {
            console.log("\n📝 Compressing research findings...");

            const compressionPrompt = compress_research_system_prompt
                .replace("{date}", date);
            const humanMessage = compress_research_human_message
                .replace("{research_topic}", researchTopic);

            const compressedResponse = await compressModel.invoke([
                new HumanMessage(compressionPrompt),
                ...messages,
                new HumanMessage(humanMessage)
            ]);

            console.log("\n--- Final Research Summary ---");
            console.log(compressedResponse.content);
            break;
        }
    }

    if (toolCallIterations >= maxIterations) {
        console.log("\n⚠️  Maximum tool iterations reached. Providing partial Results...");
    }
}

run().catch(console.error);
