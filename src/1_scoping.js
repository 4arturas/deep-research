import { StateGraph, START, END, Command, Annotation, MemorySaver } from '@langchain/langgraph';
import { HumanMessage, AIMessage, getBufferString } from '@langchain/core/messages';
import { ChatOllama } from '@langchain/ollama';
import dayjs from 'dayjs';
import { z } from 'zod';
import {MODEL_QWEN} from "./constants.js";
import { clarify_with_user_instructions, transform_messages_into_research_topic_prompt } from "./prompts.js";

const model = new ChatOllama({
    model: MODEL_QWEN,
    temperature: 0.0,
    // Some versions of Ollama support this to force pure JSON
    format: "json"
});

const AgentStateAnnotation = Annotation.Root({
    messages: Annotation({
        reducer: (x, y) => x.concat(y),
        default: () => [],
    }),
    research_brief: Annotation(),
    supervisor_messages: Annotation(),
});

const getTodayStr = () => dayjs().format('ddd MMM DD, YYYY');

async function clarify_with_user(state) {
    const date = getTodayStr();
    const messages = getBufferString(state.messages);

    const prompt = clarify_with_user_instructions
        .replace("{date}", date)
        .replace("{messages}", messages);

    const structuredLlm = model.withStructuredOutput(z.object({
        need_clarification: z.boolean(),
        question: z.string(),
        verification: z.string()
    }));

    const response = await structuredLlm.invoke([
        new HumanMessage(prompt)
    ]);

    if (response.need_clarification) {
        return new Command({
            goto: END,
            update: {
                messages: [new AIMessage(response.question)]
            }
        });
    }
    return new Command({
        goto: "write_research_brief",
        update: {
            messages: [new AIMessage(response.verification)]
        }
    });
}

async function write_research_brief(state) {
    const date = getTodayStr();
    const messages = getBufferString(state.messages);
    const prompt = transform_messages_into_research_topic_prompt
        .replace("{date}", date)
        .replace("{messages}", messages);

    const structuredLlm = model.withStructuredOutput(z.object({
        research_brief: z.string()
    }));

    const response = await structuredLlm.invoke([
        new HumanMessage(prompt)
    ]);

    return new Command({
        goto: END,
        update: {
            research_brief: response.research_brief,
            supervisor_messages: [new AIMessage(response.research_brief)]
        }
    });
}

const memory = new MemorySaver();
const graph = new StateGraph(AgentStateAnnotation)
    .addNode("clarify_with_user", clarify_with_user, {
        ends: ["write_research_brief", END]
    })
    .addNode("write_research_brief", write_research_brief)
    .addEdge(START, "clarify_with_user")
    .addEdge("write_research_brief", END)
    .compile({
        checkpointer: memory
    });


async function run(input) {
    // Convert input messages to the appropriate message types
    const convertedInput = {
        messages: input.messages.map(msg =>
            msg.type === 'human' ? new HumanMessage(msg.content) : new AIMessage(msg.content)
        )
    };

    const result = await graph.invoke(convertedInput, {
        configurable: { thread_id: `scoping-${Date.now()}` }
    });

    return result;
}

async function runLocalScopingEvaluation(graph) {
    const testCases = [
        {
            name: "Coffee Quality Research",
            input: {
                messages: [
                    { type: 'human', content: 'I want to research the best coffee shops in San Francisco.' },
                    { type: 'ai', content: 'Could you clarify what criteria are most important for determining the "best" coffee shops?' },
                    { type: 'human', content: 'Let\'s examine coffee quality to assess the best coffee shops in San Francisco.' }
                ]
            }
        },
        {
            name: "Price-focused Research",
            input: {
                messages: [
                    { type: 'human', content: 'I want to find affordable apartments in New York under $2000.' },
                    { type: 'ai', content: 'I have sufficient information to proceed. You\'re requesting research on affordable apartments in New York under $2000. I will now begin researching this topic.' }
                ]
            }
        }
    ];

    for (const testCase of testCases) {
        try {
            const convertedInput = {
                messages: testCase.input.messages.map(msg =>
                    msg.type === 'human' ? new HumanMessage(msg.content) : new AIMessage(msg.content)
                )
            };

            const result = await graph.invoke(convertedInput, {
                configurable: { thread_id: `eval-${Date.now()}` }
            });

            console.log(`Test: ${testCase.name}`);
            console.log(`Brief: ${result.research_brief || "Pending clarification..."}\n`);
        } catch (error) {
            console.error(`Error in ${testCase.name}:`, error);
        }
    }
}
// Export the run function for use in other modules
export { run };

// Example usage of the run function
async function exampleRun() {
    console.log("Running example scoping workflow...");

    // Example 1: Initial research request that needs clarification
    const result1 = await run({
        messages: [
            { type: 'human', content: 'I want to research the best coffee shops in San Francisco.' }
        ]
    });

    console.log("Result 1:", result1);

    // Example 2: Follow-up with clarification
    const result2 = await run({
        messages: [
            { type: 'human', content: 'I want to research the best coffee shops in San Francisco.' },
            { type: 'ai', content: 'Could you clarify what criteria are most important for determining the "best" coffee shops?' },
            { type: 'human', content: 'Let\'s examine coffee quality to assess the best coffee shops in San Francisco.' }
        ]
    });

    console.log("Result 2:", result2);
}

// Run the example
exampleRun().catch(console.error);