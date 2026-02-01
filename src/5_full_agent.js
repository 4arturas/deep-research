import { HumanMessage, ToolMessage, SystemMessage, AIMessage } from '@langchain/core/messages';
import { ChatOllama } from '@langchain/ollama';
import dayjs from 'dayjs';
import { z } from 'zod';
import { MODEL_QWEN } from "./constants.js";
import {
    research_agent_prompt,
    compress_research_system_prompt,
    compress_research_human_message,
    lead_researcher_prompt,
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    final_report_generation_prompt
} from "./prompts.js";
import { tool } from '@langchain/core/tools';
import * as fs from 'fs';
import * as readline from 'readline';

const getTodayStr = () => dayjs().format('ddd MMM DD, YYYY');

// ─── UTILS ─────────────────────────────────────────────────────────────

async function askUser(question) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    return new Promise(resolve => {
        rl.question(question + " ", (answer) => {
            rl.close();
            resolve(answer);
        });
    });
}

// ─── TOOLS (SUB-AGENT & SUPERVISOR) ────────────────────────────────────

// Mock Tavily Search Tool for Sub-Agents
const tavily_search = tool(
    async ({ query }) => {
        console.log(`\n  [Sub-Agent] 🔎 Searching: "${query}"`);
        // Mocking search with local file if available
        const filePath = 'C:\\Users\\4artu\\IdeaProjects\\deep-research\\src\\files\\coffee_shops_sf.md';
        try {
            if (fs.existsSync(filePath)) {
                // Return generic content if query seems related to the file
                return fs.readFileSync(filePath, 'utf8').substring(0, 1000) + "...";
            }
            return "No local mock file found. In production, this would call Tavily API.";
        } catch (e) {
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

// Think Tool
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

// Research Complete Tool
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

// ─── SUB-AGENT LOGIC ───────────────────────────────────────────────────

async function conductResearch(topic) {
    console.log(`\n🚀 [Sub-Agent] Starting research on: "${topic}"`);

    const subAgentTools = [tavily_search, think_tool];
    const subAgentToolsByName = Object.fromEntries(subAgentTools.map(t => [t.name, t]));

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
            messages.push(new ToolMessage({
                tool_call_id: toolCall.id,
                content: result,
                name: toolCall.name
            }));
        }
    }

    // Compress
    console.log(`\n📝 [Sub-Agent] Compressing findings...`);
    const date = getTodayStr();
    const compressionPrompt = compress_research_system_prompt.replace("{date}", date);
    const humanMessage = compress_research_human_message.replace("{research_topic}", topic);

    const compressedResponse = await compressModel.invoke([
        new SystemMessage(compressionPrompt),
        ...messages,
        new HumanMessage(humanMessage)
    ]);

    return compressedResponse.content;
}

const conduct_research_tool = tool(
    async ({ research_topic }) => {
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

// ─── 1. CLARIFICATION STAGE ────────────────────────────────────────────

async function verifyRequest(model, userRequest) {
    console.log("\n--- CLARIFICATION STAGE ---\n");
    const messages = [new HumanMessage(userRequest)];

    // Structured output for clarification analysis
    const clarificationSchema = z.object({
        need_clarification: z.boolean(),
        question: z.string().optional(),
        verification: z.string().optional()
    });

    const structuredModel = model.withStructuredOutput(clarificationSchema);

    while (true) {
        const date = getTodayStr();
        const messageHistoryText = messages.map(m => `${m._getType()}: ${m.content}`).join("\n");
        const formattedPrompt = clarify_with_user_instructions
            .replace("{date}", date)
            .replace("{messages}", messageHistoryText);

        const response = await structuredModel.invoke([
            new SystemMessage(formattedPrompt),
        ]);

        if (response.need_clarification) {
            console.log(`\n🤖 Question: ${response.question}`);
            const userResponse = await askUser(">> Answer (or press Enter to skip):");
            messages.push(new AIMessage(response.question)); // Model asked
            messages.push(new HumanMessage(userResponse));      // User answered
            console.log(`User: ${userResponse}`);
        } else {
            console.log(`\n✅ Verified: ${response.verification}`);
            return messages; // Return whole history
        }
    }
}

// ─── 2. BRIEF GENERATION STAGE ─────────────────────────────────────────

async function createBrief(model, messages) {
    console.log("\n--- BRIEF GENERATION STAGE ---\n");
    const date = getTodayStr();
    // Flatten messages to text for the prompt
    const messageHistoryText = messages.map(m => `${m._getType()}: ${m.content}`).join("\n");

    const formattedPrompt = transform_messages_into_research_topic_prompt
        .replace("{date}", date)
        .replace("{messages}", messageHistoryText);

    const response = await model.invoke([
        new SystemMessage(formattedPrompt)
    ]);

    const brief = response.content;
    console.log(`📋 Research Brief:\n${brief}\n`);
    return brief;
}

// ─── 3. RESEARCH STAGE (SUPERVISOR) ────────────────────────────────────

async function runResearch(model, brief) {
    console.log("\n--- RESEARCH STAGE ---\n");
    const supervisorTools = [conduct_research_tool, research_complete, think_tool];
    const supervisorToolsByName = Object.fromEntries(supervisorTools.map(t => [t.name, t]));
    const modelWithTools = model.bindTools(supervisorTools);

    let messages = [new HumanMessage(brief)];
    let iterations = 0;
    const maxResearchIterations = 10;
    const maxConcurrentResearchers = 3;
    let researchFindings = [];

    while (iterations < maxResearchIterations) {
        const date = getTodayStr();
        const systemPrompt = lead_researcher_prompt
            .replace("{date}", date)
            .replace("{max_concurrent_research_units}", maxConcurrentResearchers)
            .replace("{max_researcher_iterations}", maxResearchIterations);

        console.log(`\n👑 [Supervisor] Iteration ${iterations + 1}`);

        const response = await modelWithTools.invoke([
            new SystemMessage(systemPrompt),
            ...messages
        ]);

        messages.push(response);
        const toolCalls = response.tool_calls || [];

        if (toolCalls.length === 0) {
            console.log("\n👑 [Supervisor] No tools called. Ending loop.");
            break;
        }

        iterations++;
        let shouldStop = false;

        const toolPromises = toolCalls.map(async (toolCall) => {
            const tool = supervisorToolsByName[toolCall.name];
            if (!tool) {
                return new ToolMessage({ tool_call_id: toolCall.id, content: "Error: Tool not found", name: toolCall.name });
            }
            if (toolCall.name === "ResearchComplete") {
                shouldStop = true;
                return new ToolMessage({ tool_call_id: toolCall.id, content: "Research Completed.", name: toolCall.name });
            }

            console.log(`\n👑 [Supervisor] Calling ${toolCall.name}...`);
            let result;
            try {
                result = await tool.invoke(toolCall.args);
            } catch (e) {
                result = `Error: ${e.message}`;
            }

            // Collect findings if it was a research call
            if (toolCall.name === "ConductResearch") {
                researchFindings.push(result);
            }

            return new ToolMessage({ tool_call_id: toolCall.id, content: result, name: toolCall.name });
        });

        const toolMessages = await Promise.all(toolPromises);
        messages.push(...toolMessages);

        if (shouldStop) break;
    }

    return researchFindings.join("\n\n");
}

// ─── 4. REPORT GENERATION STAGE ────────────────────────────────────────

async function generateReport(model, researchBrief, findings) {
    console.log("\n--- REPORT GENERATION STAGE ---\n");
    const date = getTodayStr();
    const formattedPrompt = final_report_generation_prompt
        .replace("{date}", date)
        .replace("{research_brief}", researchBrief)
        .replace("{findings}", findings);

    const response = await model.invoke([
        new HumanMessage(formattedPrompt) // The prompt says "create a ... answer", so HumanMessage fits well.
    ]);

    return response.content;
}

// ─── MAIN ──────────────────────────────────────────────────────────────

async function main() {
    console.log("🚀 Starting Full Research Agent...");

    const model = new ChatOllama({ model: MODEL_QWEN, temperature: 0.0 });

    // 0. Get User Input (Hardcoded or args for now to allow simple run)
    const initialRequest = "I want to research coffee shops in San Francisco."; // Default

    // 1. Clarify
    const clarifiedMessages = await verifyRequest(model, initialRequest);

    // 2. Brief
    const researchBrief = await createBrief(model, clarifiedMessages);

    // 3. Research
    const allFindings = await runResearch(model, researchBrief);

    if (!allFindings || allFindings.trim().length === 0) {
        console.log("No findings gathered. Exiting.");
        return;
    }

    // 4. Report
    console.log("Generating final report...");
    const finalReport = await generateReport(model, researchBrief, allFindings);

    console.log("\n\n════════ FINAL REPORT ════════\n");
    console.log(finalReport);
    console.log("\n══════════════════════════════\n");

    // Save report
    fs.writeFileSync('final_report.md', finalReport);
    console.log("Report saved to final_report.md");
}

main().catch(console.error);
