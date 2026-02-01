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
import { tool_selection_prompt } from "./prompts_manual_tool.js";
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
function sanitizeInput(input) {
    if (typeof input === 'object' && input !== null) {
        return JSON.stringify(input);
    }
    return input;
}
function getToolDescriptions(tools) {
    return tools.map(t => `- ${t.toolName || t.name}: ${t.description}\n  Usage: ${t.usage}`).join("\n");
}

async function selectTool(model, tools, messages, systemPrompt) {
    const toolDescriptions = getToolDescriptions(tools);
    const formattedPrompt = tool_selection_prompt.replace("{tool_descriptions}", toolDescriptions);

    const selectorSchema = z.object({
        selected_tool: z.string().describe("The name of the tool to select. Must match one of the available tool names exactly."),
        tool_args: z.record(z.any()).describe("The arguments to pass to the tool, matching the tool's schema."),
        thought: z.string().describe("Reasoning for selecting this tool.") // Helping the model think
    });

    const selectorModel = model.withStructuredOutput(selectorSchema);

    // Combine system prompt, history, and the selection prompt
    const inputMessages = [
        new SystemMessage(systemPrompt),
        ...messages,
        new HumanMessage(formattedPrompt) // Explicit instruction at the end
    ];

    const response = await selectorModel.invoke(inputMessages);

    return response;
}

// ─── TOOLS (SUB-AGENT & SUPERVISOR) ────────────────────────────────────

async function tavily_search({ query }) {
    query = sanitizeInput(query);
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
}
tavily_search.toolName = "tavily_search";
tavily_search.description = "Search for information on the web.";
tavily_search.usage = "{ query: string }";


async function think_tool({ reflection }) {
    reflection = sanitizeInput(reflection);
    console.log(`\n💭 Reflection: ${reflection}`);
    return `Reflection recorded: ${reflection}`;
}
think_tool.toolName = "think_tool";
think_tool.description = "Strategic planning and reflection tool.";
think_tool.usage = "{ reflection: string }";

async function research_complete() {
    console.log("\n✅ Supervisor decided research is complete.");
    return "RESEARCH_COMPLETE_SIGNAL";
}
research_complete.toolName = "ResearchComplete";
research_complete.description = "Indicate that research is complete.";
research_complete.usage = "{}";


// ─── SUB-AGENT LOGIC ───────────────────────────────────────────────────

async function conductResearch(topic) {
    if (typeof topic === 'object' && topic !== null) {
        topic = topic.research_topic || JSON.stringify(topic);
    }
    topic = sanitizeInput(topic);
    console.log(`\n🚀 [Sub-Agent] Starting research on: "${topic}"`);

    const subAgentTools = [tavily_search, think_tool];

    const model = new ChatOllama({ model: MODEL_QWEN, temperature: 0.0 });
    const compressModel = new ChatOllama({ model: MODEL_QWEN, temperature: 0.0 });

    const messages = [new HumanMessage(topic)];
    let iterations = 0;
    const maxIterations = 5;

    while (iterations < maxIterations) {
        const date = getTodayStr();
        const systemPrompt = research_agent_prompt.replace("{date}", date);

        // MANUAL TOOL SELECTION
        let selection;
        try {
            selection = await selectTool(model, subAgentTools, messages, systemPrompt);
        } catch (e) {
            console.error("Tool selection failed:", e);
            break;
        }

        if (!selection || !selection.selected_tool) {
            break;
        }

        if (selection.thought) {
            console.log(`\n[Model Thought]: ${selection.thought}`);
        }

        const toolName = selection.selected_tool;
        const toolArgs = selection.tool_args || {};

        console.log(`[Selected Tool]: ${toolName}`);

        iterations++;

        let result;
        try {
            if (toolName === tavily_search.toolName) {
                result = await tavily_search(toolArgs);
            } else if (toolName === think_tool.toolName) {
                result = await think_tool(toolArgs);
            } else {
                console.error(`Tool ${toolName} not found.`);
                messages.push(new AIMessage(`Error: Tool ${toolName} not found.`));
                continue;
            }
        } catch (e) {
            result = `Error: ${e.message}`;
        }

        messages.push(new AIMessage({
            content: selection.thought || "",
            tool_calls: [{
                name: toolName,
                args: toolArgs,
                id: `call_${Date.now()}`
            }]
        }));

        messages.push(new ToolMessage({
            tool_call_id: `call_${Date.now()}`,
            content: result,
            name: toolName
        }));
    }

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

async function conduct_research_tool({ research_topic }) {
    return await conductResearch(research_topic);
}
conduct_research_tool.toolName = "ConductResearch";
conduct_research_tool.description = "Delegate research tasks to specialized sub-agents.";
conduct_research_tool.usage = "{ research_topic: string }";


// ─── 1. CLARIFICATION STAGE ────────────────────────────────────────────

async function verifyRequest(model, userRequest) {
    console.log("\n--- CLARIFICATION STAGE ---\n");
    const messages = [new HumanMessage(userRequest)];

    const structuredModel = model.withStructuredOutput(z.object({
        need_clarification: z.boolean(),
        question: z.string().optional(),
        verification: z.string().optional()
    }));

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
            messages.push(new AIMessage(response.question));    // Model asked
            messages.push(new HumanMessage(userResponse));      // User answered
            console.log(`User: ${userResponse}`);
        } else {
            console.log(`\n✅ Verified: ${response.verification}`);
            return messages;
        }
    }
}

// ─── 2. BRIEF GENERATION STAGE ─────────────────────────────────────────

async function createBrief(model, messages) {
    console.log("\n--- BRIEF GENERATION STAGE ---\n");
    const date = getTodayStr();
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

        let selection;
        try {
            selection = await selectTool(model, supervisorTools, messages, systemPrompt);
        } catch (e) {
            console.error("Supervisor tool selection failed:", e);
            break;
        }

        if (!selection || !selection.selected_tool) {
            console.log("\n👑 [Supervisor] No tool selected. Ending loop.");
            break;
        }

        if (selection.thought) {
            console.log(`\n[Supervisor Thought]: ${selection.thought}`);
        }

        const toolName = selection.selected_tool;
        const toolArgs = selection.tool_args || {};
        const runId = `call_${Date.now()}`;

        // Add "AI Message" to history
        messages.push(new AIMessage({
            content: selection.thought || "",
            tool_calls: [{
                name: toolName,
                args: toolArgs,
                id: runId
            }]
        }));

        iterations++;

        if (toolName === research_complete.toolName) {
            console.log("\n✅ Supervisor decided research is complete.");
            break;
        }

        let result;
        try {
            if (toolName === conduct_research_tool.toolName) {
                result = await conduct_research_tool(toolArgs);
            } else if (toolName === think_tool.toolName) {
                result = await think_tool(toolArgs);
            } else {
                console.error(`Tool ${toolName} not found.`);
                messages.push(new ToolMessage({ tool_call_id: runId, content: "Error: Tool not found", name: toolName }));
                continue;
            }
        } catch (e) {
            result = `Error: ${e.message}`;
        }

        if (toolName === conduct_research_tool.toolName) {
            researchFindings.push(result);
        }

        messages.push(new ToolMessage({ tool_call_id: runId, content: result, name: toolName }));
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
    console.log("🚀 Starting Full Research Agent (Manual Tool Selection + Plain Functions)...");

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
    fs.writeFileSync('final_report_manual.md', finalReport);
    console.log("Report saved to final_report_manual.md");
}

main().catch(console.error);
