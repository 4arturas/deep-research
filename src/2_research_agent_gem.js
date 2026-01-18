import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { StateGraph, START, END, Command, Annotation } from '@langchain/langgraph';
import {
    HumanMessage,
    AIMessage,
    getBufferString,
    ToolMessage,
} from '@langchain/core/messages';
import { ChatOllama } from '@langchain/ollama';
import { tool } from '@langchain/core/tools';
import dayjs from 'dayjs';
import { z } from 'zod';
import { MODEL_QWEN } from "./constants.js";
import {
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt
} from "./prompts.js";

// Human interaction tool
const askUser = tool(
    async ({ question }) => {
        const rl = readline.createInterface({ input, output });
        console.log(`\nAI: ${question}`);
        const answer = await rl.question('You: ');
        rl.close();
        return answer;
    },
    {
        name: "ask_user",
        description: "Ask the user for clarification or additional information",
        schema: z.object({
            question: z.string().describe("Question to ask the user")
        })
    }
);

// Model with tools bound
const model = new ChatOllama({
    model: MODEL_QWEN,
    temperature: 0.0
}).bindTools([askUser]);

const getTodayStr = () => dayjs().format('ddd MMM DD, YYYY');

const AgentStateNotion = Annotation.Root({
    messages: Annotation({
        reducer: (x, y) => x.concat(y),
        default: () => []
    }),
    supervisor_messages: Annotation(),
    research_brief: Annotation()
});

async function clarify_with_user(state) {
    const date = getTodayStr();
    const messages = getBufferString(state.messages);
    
    // System prompt for the AI
    const systemPrompt = `You are a research assistant helping to clarify user requests. 
Today's date is ${date}.

Analyze the user's request and determine if you need clarification:
- If you need more information, use the ask_user tool to ask a specific question
- If you have enough information, provide a verification message and proceed to research

User messages so far:
${messages}

Guidelines:
- Ask concise, focused questions
- Don't ask for information already provided
- Focus on the most important missing details`;

    const response = await model.invoke([
        new HumanMessage(systemPrompt)
    ]);

    // Check if AI wants to ask user a question
    if (response.tool_calls && response.tool_calls.length > 0) {
        const toolCall = response.tool_calls[0];
        if (toolCall.name === "ask_user") {
            // Execute the tool to get user input
            const userAnswer = await askUser.invoke(toolCall.args);
            
            return new Command({
                goto: "clarify_with_user", // Loop back to process user's answer
                update: {
                    messages: [
                        response,
                        new ToolMessage({
                            name: toolCall.name,
                            tool_call_id: toolCall.id,
                            content: userAnswer
                        }),
                        new HumanMessage(userAnswer)
                    ]
                }
            });
        }
    }

    // AI has enough information, proceed to research
    return new Command({
        goto: "write_research_brief",
        update: {
            messages: [response]
        }
    });
}

async function write_research_brief(state) {
    const date = getTodayStr();
    const messages = getBufferString(state.messages);
    const prompt = transform_messages_into_research_topic_prompt
        .replace("{date}", date)
        .replace("{messages}", messages);
    
    // Use structured output for consistent format
    const structLlm = model.withStructuredOutput(z.object({
        research_brief: z.string()
    }));
    
    const response = await structLlm.invoke([
        new HumanMessage(prompt)
    ]);
    
    console.log("\n--- Research Brief Generated ---");
    console.log(response.research_brief);
    
    return new Command({
        goto: END,
        update: {
            research_brief: String(response.research_brief),
            supervisor_messages: [new AIMessage(response.research_brief)]
        }
    });
}

async function run() {
    const rl = readline.createInterface({ input, output });

    const graph = new StateGraph(AgentStateNotion)
        .addNode("clarify_with_user", clarify_with_user, { ends: ["write_research_brief", END] })
        .addNode("write_research_brief", write_research_brief, { ends: [END] })
        .addEdge(START, "clarify_with_user")
        .compile();

    let currentState = {
        messages: [new HumanMessage("I want to go for a vocation.")]
    };

    while (true) {
        const response = await graph.invoke(currentState);
        currentState.messages = response.messages;

        if (response.research_brief) {
            console.log("\n--- Final Research Brief ---");
            console.log(response.research_brief);
            break;
        }

        const lastMessage = response.messages[response.messages.length - 1];
        if (lastMessage instanceof AIMessage) {
            console.log(`\nAI: ${lastMessage.content}`);
            const userInput = await rl.question('You: ');
            currentState.messages.push(new HumanMessage(userInput));
        }
    }

    rl.close();
}

run();