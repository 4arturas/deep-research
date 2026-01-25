import { HumanMessage, ToolMessage, SystemMessage } from '@langchain/core/messages';
import { ChatOllama } from '@langchain/ollama';
import dayjs from 'dayjs';
import { z } from 'zod';
import { MODEL_QWEN } from "./constants.js";
import {
    research_agent_prompt,
    compress_research_system_prompt,
    compress_research_human_message,
    lead_researcher_prompt
} from "./prompts.js";
import { tool } from '@langchain/core/tools';
import * as fs from 'fs';

const getTodayStr = () => dayjs().format('ddd MMM DD, YYYY');


const tavily_search = tool(
    async ({ query }) => {
        console.log(`\n  [Sub-Agent] 🔎 Searching: "${query}"`);
        // Mocking search with local file if available
        const filePath = 'C:\\Users\\4artu\\IdeaProjects\\deep-research\\src\\files\\coffee_shops_sf.md';
        try {
            if (fs.existsSync(filePath)) {
                const rawContent = fs.readFileSync(filePath, 'utf8');
                const maxChars = 800;
                if (rawContent.length <= maxChars) return rawContent;
                return rawContent.substring(0, maxChars) + "\n\n[Truncated...]";
            }
            return "No local mock file found. In production, this would call Tavily API.";
        } catch (e) {
            console.error("  [Sub-Agent] Error reading file:", e);
            return "Error reading search results.";
        }
    },
    {
        name: "tavily_search",
        description: "Search for information on the web.",
        schema: z.object({
            query: z.string().describe("The search query")
        })
    }
);

// Think Tool for both Supervisor and Sub-Agents
const think_tool = tool(
    async ({ reflection }) => {
        console.log(`\n💭 Reflection: ${reflection}`);
        return `Reflection recorded: ${reflection}`;
    },
    {
        name: "think_tool",
        description: "Strategic planning and reflection tool.",
        schema: z.object({
            reflection: z.string().describe("Reflection content")
        })
    }
);

// Research Complete Tool for Supervisor
const research_complete = tool(
    async () => {
        console.log("\n✅ Supervisor decided research is complete.");
        return "RESEARCH_COMPLETE_SIGNAL";
    },
    {
        name: "ResearchComplete",
        description: "Indicate that research is complete.",
        schema: z.object({})
    }
);

// ─── SUB-AGENT ─────────────────────────────────────────────────────────

async function conductResearch(topic) {
    console.log(`\n🚀 [Sub-Agent] Starting research on: "${topic}"`);

    // Sub-agent specific tools and model
    const subAgentTools = [tavily_search, think_tool];
    const subAgentToolsByName = Object.fromEntries(
        subAgentTools.map(t => [t.name, t])
    );

    const model = new ChatOllama({ model: MODEL_QWEN, temperature: 0.0 });
    const modelWithTools = model.bindTools(subAgentTools);
    const compressModel = new ChatOllama({ model: MODEL_QWEN, temperature: 0.0 });

    const messages = [new HumanMessage(topic)];
    let iterations = 0;
    const maxIterations = 5;

    while (iterations < maxIterations) {
        const date = getTodayStr();
        const systemPrompt = research_agent_prompt.replace("{date}", date);

        const response = await modelWithTools.invoke([
            new SystemMessage(systemPrompt),
            ...messages
        ]);

        messages.push(response);

        const toolCalls = response.tool_calls || [];
        // If no tools called, we treat it as a final response, but we prefer compression.
        if (toolCalls.length === 0) {
            break;
        }
        iterations++;

        for (const toolCall of toolCalls) {
            const tool = subAgentToolsByName[toolCall.name];
            let result;
            try {
                result = await tool.invoke(toolCall.args);
            } catch (e) {
                result = `Error: ${e.message}`;
            }

            const toolMessage = new ToolMessage({
                name: toolCall.name,
                tool_call_id: toolCall.id,
                content: result
            });
            messages.push( toolMessage );
        } // end for toolCall
    } // end while

    // Compress findings
    console.log(`\n📝 [Sub-Agent] Compressing findings for: "${topic}"...`);
    const date = getTodayStr();
    const compressionPrompt = compress_research_system_prompt.replace("{date}", date);
    const humanMessage = compress_research_human_message.replace("{research_topic}", topic);

    // Filter messages for compression (exclude think_tool as per guidelines)
    // We'll pass the whole history but the prompt says to focus on substantive content.
    // The previous implementation passed all messages. We'll stick to that but maybe clarify the prompt allows it.
    // Actually, the prompt says "Tool Call Filtering... Exclude think_tool calls".
    // Let's do a simple filter if needed, but the model instructions are usually enough.
    // For safety, let's just pass everything, as the prompt instruction is for the model to filter.

    const compressedResponse = await compressModel.invoke([
        new SystemMessage(compressionPrompt),
        ...messages,
        new HumanMessage(humanMessage)
    ]);

    console.log(`\n  [Sub-Agent] Report ready.`);
    return compressedResponse.content;
}

// Conduct Research Tool for Supervisor (Wraps the sub-agent)
const conduct_research_tool = tool(
    async ({ research_topic }) => {
        // Run the sub-agent
        return await conductResearch(research_topic);
    },
    {
        name: "ConductResearch",
        description: "Delegate research tasks to specialized sub-agents.",
        schema: z.object({
            research_topic: z.string().describe("The topic to research.")
        })
    }
);

// ─── SUPERVISOR ────────────────────────────────────────────────────────

async function runSupervisor() {
    console.log("🌟 Starting Research Supervisor...");

    const supervisorTools = [conduct_research_tool, research_complete, think_tool];
    const supervisorToolsByName = Object.fromEntries(supervisorTools.map(t => [t.name, t]));

    const model = new ChatOllama({ model: MODEL_QWEN, temperature: 0.0 });
    const modelWithTools = model.bindTools(supervisorTools);

    // Initial user request
    const userRequest = "I want to research coffee shops in San Francisco.";
    console.log(`❓ User Request: "${userRequest}"\n`);

    // Supervisor State
    let messages = [new HumanMessage(userRequest)];
    let iterations = 0;
    const maxResearcherIterations = 10;
    const maxConcurrentResearchUnits = 3;

    while (iterations < maxResearcherIterations) {
        const date = getTodayStr();
        const systemPrompt = lead_researcher_prompt
            .replace("{date}", date)
            .replace("{max_concurrent_research_units}", maxConcurrentResearchUnits)
            .replace("{max_researcher_iterations}", maxResearcherIterations);

        console.log(`\n👑 [Supervisor] Iteration ${iterations + 1}`);

        const response = await modelWithTools.invoke([
            new SystemMessage(systemPrompt),
            ...messages
        ]);

        messages.push(response);
        const toolCalls = response.tool_calls || [];

        if (toolCalls.length === 0) {
            console.log("\n👑 [Supervisor] No tools called. Ending.");
            // If the model creates a final text response without calling ResearchComplete, strictly typically we might want to force it, 
            // but here we'll just accept it as the final output.
            console.log("Response:", response.content);
            break;
        }

        iterations++;

        let shouldStop = false;

        // Execute tools (ConductResearch in parallel if needed)
        // We'll collect promises for ConductResearch and await them together
        const toolPromises = toolCalls.map(async (toolCall) => {
            const tool = supervisorToolsByName[toolCall.name];
            if (!tool) {
                return new ToolMessage({
                    tool_call_id: toolCall.id,
                    content: "Error: Tool not found",
                    name: toolCall.name
                });
            }

            // Special handling for ResearchComplete
            if (toolCall.name === "ResearchComplete") {
                shouldStop = true;
                return new ToolMessage({
                    tool_call_id: toolCall.id,
                    content: "Research Completed.",
                    name: toolCall.name
                });
            }

            console.log(`\n👑 [Supervisor] Calling ${toolCall.name}...`);
            let result;
            try {
                result = await tool.invoke(toolCall.args);
            } catch (e) {
                result = `Error: ${e.message}`;
            }

            return new ToolMessage({
                tool_call_id: toolCall.id,
                content: result,
                name: toolCall.name
            });
        });

        const toolMessages = await Promise.all(toolPromises);
        messages.push(...toolMessages);

        if (shouldStop) {
            console.log("\n🎉 Research Process Completed!");
            break;
        }
    } // end while

    // Final Summary / Output
    // The notebook usually compiles the notes. Since our mock supervisor loop just ends,
    // we can print the final accumulated notes or the last response.
    // In strict following of the notebook, we might want to generate a final report using `final_report_generation_prompt`.
    // But for this conversion, ensuring the "Supervisor delegates" flow is the main goal.
    // Let's grab the research findings from the history and show them.

    console.log("\n📊 --- Final Research State ---");
    const researchFindings = messages
        .filter(m => m instanceof ToolMessage && m.name === "ConductResearch")
        .map(m => m.content)
        .join("\n\n---\n\n");

    if (researchFindings) {
        console.log(researchFindings);
    } else {
        console.log("No research findings were collected.");
    }
}

runSupervisor().catch(console.error);
