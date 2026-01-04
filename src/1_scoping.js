import { StateGraph, START, END, Command, Annotation, MemorySaver } from '@langchain/langgraph';
import { HumanMessage, AIMessage, getBufferString } from '@langchain/core/messages';
import { ChatOllama } from '@langchain/ollama';
import dayjs from 'dayjs';
import { z } from 'zod';
import {MODEL_GEMMA_3_27b_CLOUD, MODEL_MINISTRAL_3_14B_CLOUD} from "./constants.js";

const clarifyWithUserInstructions = `These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

Today's date is {date}.

Assess whether you need to ask a clarifying question, or if the user has already provided enough information for you to start research.
IMPORTANT: If you can see in the messages history that you have already asked a clarifying question, you almost always do not need to ask another one. Only ask another question if ABSOLUTELY NECESSARY.

If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.
If you need to ask a question, follow these guidelines:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
- Use bullet points or numbered lists if appropriate for clarity. Make sure that this uses markdown formatting and will be rendered correctly if the string output is passed to a markdown renderer.
- Don't ask for unnecessary information, or information that the user has already provided. If you can see that the user has already provided the information, do not ask for it again.

Respond in valid JSON format with these exact keys:
"need_clarification": boolean,
"question": "<question to ask the user to clarify the report scope>",
"verification": "<verification message that we will start research>"

If you need to ask a clarifying question, return:
"need_clarification": true,
"question": "<your clarifying question>",
"verification": ""

If you do not need to ask a clarifying question, return:
"need_clarification": false,
"question": "",
"verification": "<acknowledgement message that you will now start research based on the provided information>"

For the verification message when no clarification is needed:
- Acknowledge that you have sufficient information to proceed
- Briefly summarize the key aspects of what you understand from their request
- Confirm that you will now begin the research process
- Keep the message concise and professional`;

const transformMessagesIntoResearchTopicPrompt = `You will be given a set of messages that have been exchanged so far between yourself and the user.
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.

The messages that have been exchanged so far between yourself and the user are:
<Messages>
{messages}
</Messages>

Today's date is {date}.

You will return a single research question that will be used to guide the research.

Guidelines:
1. Maximize Specificity and Detail
- Include all known user preferences and explicitly list key attributes or dimensions to consider.
- It is important that all details from the user are included in the instructions.

2. Handle Unstated Dimensions Carefully
- When research quality requires considering additional dimensions that the user hasn't specified, acknowledge them as open considerations rather than assumed preferences.
- Example: Instead of assuming "budget-friendly options," say "consider all price ranges unless cost constraints are specified."
- Only mention dimensions that are genuinely necessary for comprehensive research in that domain.

3. Avoid Unwarranted Assumptions
- Never invent specific user preferences, constraints, or requirements that weren't stated.
- If the user hasn't provided a particular detail, explicitly note this lack of specification.
- Guide the researcher to treat unspecified aspects as flexible rather than making assumptions.

4. Distinguish Between Research Scope and User Preferences
- Research scope: What topics/dimensions should be investigated (can be broader than user's explicit mentions)
- User preferences: Specific constraints, requirements, or preferences (must only include what user stated)
- Example: "Research coffee quality factors (including bean sourcing, roasting methods, brewing techniques) for San Francisco coffee shops, with primary focus on taste as specified by the user."

5. Use the First Person
- Phrase the request from the perspective of the user.

6. Sources
- If specific sources should be prioritized, specify them in the research question.
- For product and travel research, prefer linking directly to official or primary websites (e.g., official brand sites, manufacturer pages, or reputable e-commerce platforms like Amazon for user reviews) rather than aggregator sites or SEO-heavy blogs.
- For academic or scientific queries, prefer linking directly to the original paper or official journal publication rather than survey papers or secondary summaries.
- For people, try linking directly to their LinkedIn profile, or their personal website if they have one.
- If the query is in a specific language, prioritize sources published in that language.

Respond in valid JSON format with this exact key:
"research_brief": "<detailed research brief content>"`


const model = new ChatOllama({
  model: MODEL_GEMMA_3_27b_CLOUD,
  temperature: 0.0
});

const ClarifyWithUserSchema = z.object({
  need_clarification: z.boolean()
      .describe("Whether the user needs to be asked a clarifying question."),
  question: z.string()
      .describe("A question to ask the user to clarify the report scope."),
  verification: z.string()
      .describe("Verify message that we will start research after the user has provided the necessary information.")
});

const ResearchQuestionSchema = z.object({
  research_brief: z.string()
      .describe("A research question or brief that will be used to guide the research.")
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

async function clarifyWithUser(state) {
  const structuredOutputModel = model.withStructuredOutput(ClarifyWithUserSchema);

  const prompt = clarifyWithUserInstructions
      .replace('{messages}', getBufferString(state.messages))
      .replace('{date}', getTodayStr());

  const response = await structuredOutputModel.invoke([new HumanMessage(prompt)]);

  if (response.need_clarification) {
    return new Command({
      goto: END,
      update: { messages: [new AIMessage(response.question)] }
    });
  } else {
    return new Command({
      goto: "write_research_brief",
      update: { messages: [new AIMessage(response.verification)] }
    });
  }
}

async function writeResearchBrief(state) {
  const structuredOutputModel = model.withStructuredOutput(ResearchQuestionSchema);

  const prompt = transformMessagesIntoResearchTopicPrompt
      .replace('{messages}', getBufferString(state.messages))
      .replace('{date}', getTodayStr());

  const response = await structuredOutputModel.invoke([new HumanMessage(prompt)]);

  return {
    research_brief: response.research_brief,
    supervisor_messages: [new HumanMessage(response.research_brief)]
  };
}

const memory = new MemorySaver();
const graph = new StateGraph(AgentStateAnnotation)
    .addNode("clarify_with_user", clarifyWithUser, {
      ends: ["write_research_brief", END]
    })
    .addNode("write_research_brief", writeResearchBrief)
    .addEdge(START, "clarify_with_user")
    .addEdge("write_research_brief", END)
    .compile({
      checkpointer: memory
    });

export { graph };

async function main() {
  const config = { configurable: { thread_id: 'coffee-research-final' } };

  const inputs = [
    "I want to research the best coffee shops in San Francisco.",
    "Let's examine coffee quality to assess the best coffee shops in San Francisco."
  ];

  for (const text of inputs) {
    console.log(`Human\n${text}`);

    // Pass only the new message; the reducer + checkpointer handles the history
    await graph.invoke({ messages: [new HumanMessage(text)] }, config);

    const state = await graph.getState(config);
    const lastMsg = state.values.messages[state.values.messages.length - 1];

    console.log(`AI\n${lastMsg.content}\n`);

    if (state.values.research_brief) {
      console.log("✅ RESEARCH BRIEF GENERATED:");
      console.log(state.values.research_brief);
      break;
    }
  }
}

async function runLocalScopingEvaluation(graph) {
    console.log("=== Running Local Scoping Evaluation ===\n");

    // Define test cases
    const testCases = [
        {
            name: "Coffee Quality Research",
            input: {
                messages: [
                    { type: 'human', content: 'I want to research the best coffee shops in San Francisco.' },
                    { type: 'ai', content: 'Could you clarify what criteria are most important for determining the "best" coffee shops?' },
                    { type: 'human', content: 'Let\'s examine coffee quality to assess the best coffee shops in San Francisco.' }
                ]
            },
            expected_criteria: ["Quality requirements", "Location requirements", "Coffee-related requirements"]
        },
        {
            name: "Price-focused Research",
            input: {
                messages: [
                    { type: 'human', content: 'I want to find affordable apartments in New York under $2000.' },
                    { type: 'ai', content: 'I have sufficient information to proceed. You\'re requesting research on affordable apartments in New York under $2000. I will now begin researching this topic.' }
                ]
            },
            expected_criteria: ["Price/cost requirements", "Location requirements"]
        }
    ];

    // Run evaluation for each test case
    for (const testCase of testCases) {
        console.log(`Evaluating: ${testCase.name}`);
        console.log(`Input: ${JSON.stringify(testCase.input.messages.map(m => m.content), null, 2)}`);

        try {
            // Convert messages to proper format
            const convertedInput = {
                messages: testCase.input.messages.map(msg => {
                    if (msg.type === 'human') {
                        return new HumanMessage(msg.content);
                    } else if (msg.type === 'ai') {
                        return new AIMessage(msg.content);
                    } else {
                        return new HumanMessage(msg.content);
                    }
                })
            };

            // Run the graph
            const result = await graph.invoke(convertedInput, {
                configurable: {
                    thread_id: `eval-${Date.now()}-${Math.random()}`
                }
            });

            console.log(`Graph result:`, result);

            // Extract research brief
            const researchBrief = result.research_brief || "No research brief generated";
            console.log(`Generated brief: ${researchBrief.substring(0, 200)}...`);

            // Run evaluation functions
            const runInput = testCase.input;
            const runOutput = {
                research_brief: researchBrief,
                messages: testCase.input.messages
            };

            console.log("\nRunning success criteria evaluation...");
            const successResult = await evaluateSuccessCriteria(runInput, runOutput);
            console.log(`Success Criteria Score: ${successResult.score} (${successResult.reason})`);

            console.log("\nRunning hallucination evaluation...");
            const hallucinationResult = await evaluateHallucination(runInput, runOutput);
            console.log(`Hallucination Score: ${hallucinationResult.score} (${hallucinationResult.reason})`);

            console.log("-----\n");

        } catch (error) {
            console.error(`Error evaluating ${testCase.name}:`, error);
        }
    }

    console.log("=== Local Scoping Evaluation Complete ===");
}

// main().catch(console.error);

runLocalScopingEvaluation(graph).catch(console.error);