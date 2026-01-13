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

const ClarifyWithUserSchema = z.object({
    need_clarification: z.boolean(),
    question: z.string(),
    verification: z.string()
});

const ResearchQuestionSchema = z.object({
    research_brief: z.string()
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
    const structuredLlm = model.withStructuredOutput(ClarifyWithUserSchema);

    const date = getTodayStr();
    const messages = getBufferString(state.messages);

    const prompt = clarify_with_user_instructions
        .replace("{date}", date)
        .replace("{messages}", messages);

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
    const structuredLlm = model.withStructuredOutput(ResearchQuestionSchema);

    const date = getTodayStr();
    const messages = getBufferString(state.messages);
    const prompt = transform_messages_into_research_topic_prompt
        .replace("{date}", date)
        .replace("{messages}", messages);

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

runLocalScopingEvaluation(graph).catch(console.error);