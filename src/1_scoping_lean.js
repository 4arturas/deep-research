import { HumanMessage, AIMessage, getBufferString } from '@langchain/core/messages';
import { ChatOllama } from '@langchain/ollama';
import dayjs from 'dayjs';
import { z } from 'zod';
import { MODEL_QWEN } from "./constants.js";
import { clarify_with_user_instructions, transform_messages_into_research_topic_prompt } from "./prompts.js";
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

const model = new ChatOllama({ model: MODEL_QWEN, temperature: 0.0 });

const getTodayStr = () => dayjs().format('ddd MMM DD, YYYY');

async function run() {
    const rl = readline.createInterface({ input, output });

    const initialRequest = await rl.question('\nWhat would you like to research? ');
    let messages = [new HumanMessage(initialRequest)];

    while (true) {
        const date = dayjs().format('ddd MMM DD, YYYY');
        const messageHistory = getBufferString(messages);

        const prompt1 = clarify_with_user_instructions
            .replace("{date}", date)
            .replace("{messages}", messageHistory);
        // Determine if we need clarification or should generate brief
        const response = await model.withStructuredOutput(z.object({
            need_clarification: z.boolean(),
            question: z.string(),
            verification: z.string(),
        })).invoke([new HumanMessage(
            prompt1
        )]);

        // If AI needs clarification, ask user
        if (response.need_clarification) {
            console.log(`\nAI: ${response.question}`);
            const userInput = await rl.question('You: ');

            messages.push(new AIMessage(response.question));
            messages.push(new HumanMessage(userInput));
            continue;
        }

        // If AI has enough info, generate research brief
        if (response.verification) {
            console.log(`\nAI: ${response.verification}`);

            const prompt2 =  transform_messages_into_research_topic_prompt
                .replace("{date}", date)
                .replace("{messages}", messageHistory);
            const briefResponse = await model.withStructuredOutput(z.object({
                research_brief: z.string()
            })).invoke([new HumanMessage(
                prompt2
            )]);

            console.log("\n--- Research Brief ---");
            console.log(briefResponse.research_brief);
            break;
        }
    }

    rl.close();
}

run().catch(console.error);