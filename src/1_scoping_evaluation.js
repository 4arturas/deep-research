import { graph } from './1_scoping.js';
import { HumanMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOllama } from '@langchain/ollama';
import { z } from 'zod';

const CriteriaEvaluationSchema = z.object({
    reasoning: z.string(),
    score: z.number().int().min(0).max(1),
    criterion_evaluated: z.string(),
    evidence: z.string()
}).strict();

const HallucinationEvaluationSchema = z.object({
    reasoning: z.string(),
    score: z.number().int().min(0).max(1),
    evidence: z.string()
}).strict();

const BRIEF_CRITERIA_PROMPT = `You are an expert research brief evaluator specializing in assessing whether generated research briefs accurately capture user-specified criteria without loss of important details.

<role>
You are an expert research brief evaluator specializing in assessing whether generated research briefs accurately capture user-specified criteria without loss of important details.
</role>

<task>
Determine if the research brief adequately captures the specific success criterion provided. Return a binary assessment with detailed reasoning.
</task>

<evaluation_context>
Research briefs are critical for guiding downstream research agents. Missing or inadequately captured criteria can lead to incomplete research that fails to address user needs. Accurate evaluation ensures research quality and user satisfaction.
</evaluation_context>

<criterion_to_evaluate>
{criterion}
</criterion_to_evaluate>

<research_brief>
{research_brief}
</research_brief>

<evaluation_guidelines>
CAPTURED (criterion is adequately represented) if:
- The research brief explicitly mentions or directly addresses the criterion
- The brief contains equivalent language or concepts that clearly cover the criterion
- The criterion's intent is preserved even if worded differently
- All key aspects of the criterion are represented in the brief

NOT CAPTURED (criterion is inadequately addressed) if:
- The criterion is completely absent from the research brief
- The brief only partially addresses the criterion, missing important aspects
- The criterion is implied but not clearly stated or actionable for researchers
- The brief contradicts or conflicts with the criterion
</evaluation_guidelines>

<evaluation_examples>
Example 1 - CAPTURED:
Criterion: "Current age is 25"
Brief: "...investment advice for a 25-year-old investor..."
Judgment: CAPTURED - age is explicitly mentioned

Example 2 - NOT CAPTURED:
Criterion: "Monthly rent below 7k"
Brief: "...find apartments in Manhattan with good amenities..."
Judgment: NOT CAPTURED - budget constraint is completely missing

Example 3 - CAPTURED:
Criterion: "High risk tolerance"
Brief: "...willing to accept significant market volatility for higher returns..."
Judgment: CAPTURED - equivalent concept expressed differently

Example 4 - NOT CAPTURED:
Criterion: "Doorman building required"
Brief: "...find apartments with modern amenities..."
Judgment: NOT CAPTURED - specific doorman requirement not mentioned
</evaluation_examples>

<output_instructions>
1. Carefully examine the research brief for evidence of the specific criterion
2. Look for both explicit mentions and equivalent concepts
3. Provide specific quotes or references from the brief as evidence
4. Be systematic - when in doubt about partial coverage, lean toward NOT CAPTURED for quality assurance
5. Focus on whether a researcher could act on this criterion based on the brief alone
</output_instructions>

Return your response as a JSON object with the following structure:
{
  "reasoning": "Detailed explanation of your evaluation",
  "score": 1 if CAPTURED, 0 if NOT CAPTURED,
  "criterion_evaluated": "The original criterion text",
  "evidence": "Specific quotes or references from the brief that support your evaluation"
}`;

const BRIEF_HALLUCINATION_PROMPT = `## Brief Hallucination Evaluator

<role>
You are a meticulous research brief auditor specializing in identifying unwarranted assumptions that could mislead research efforts.
</role>

<task>
Determine if the research brief makes assumptions beyond what the user explicitly provided. Return a binary pass/fail judgment.
</task>

<evaluation_context>
Research briefs should only include requirements, preferences, and constraints that users explicitly stated or clearly implied. Adding assumptions can lead to research that misses the user's actual needs.
</evaluation_context>

<research_brief>
{research_brief}
</research_brief>

<success_criteria>
{success_criteria}
</success_criteria>

<evaluation_guidelines>
PASS (no unwarranted assumptions) if:
- Brief only includes explicitly stated user requirements
- Any inferences are clearly marked as such or logically necessary
- Source suggestions are general recommendations, not specific assumptions
- Brief stays within the scope of what the user actually requested

FAIL (contains unwarranted assumptions) if:
- Brief adds specific preferences user never mentioned
- Brief assumes demographic, geographic, or contextual details not provided
- Brief narrows scope beyond user's stated constraints
- Brief introduces requirements user didn't specify
</evaluation_guidelines>

<evaluation_examples>
Example 1 - PASS:
User criteria: ["Looking for coffee shops", "In San Francisco"]
Brief: "...research coffee shops in San Francisco area..."
Judgment: PASS - stays within stated scope

Example 2 - FAIL:
User criteria: ["Looking for coffee shops", "In San Francisco"]
Brief: "...research trendy coffee shops for young professionals in San Francisco..."
Judgment: FAIL - assumes "trendy" and "young professionals" demographics

Example 3 - PASS:
User criteria: ["Budget under $3000", "2 bedroom apartment"]
Brief: "...find 2-bedroom apartments within $3000 budget, consulting rental sites and local listings..."
Judgment: PASS - source suggestions are appropriate, no preference assumptions

Example 4 - FAIL:
User criteria: ["Budget under $3000", "2 bedroom apartment"]
Brief: "...find modern 2-bedroom apartments under $3000 in safe neighborhoods with good schools..."
Judgment: FAIL - assumes "modern", "safe", and "good schools" preferences
</evaluation_examples>

<output_instructions>
Carefully scan the brief for any details not explicitly provided by the user. Be strict - when in doubt about whether something was user-specified, lean toward FAIL.
</output_instructions>

Return your response as a JSON object with the following structure:
{
  "reasoning": "Detailed explanation of your evaluation",
  "score": 1 if PASS (no hallucination), 0 if FAIL (contains hallucination),
  "evidence": "Specific examples of hallucinated content or confirmation of no hallucination"
}`;

function extractCriteriaFromConversation(messages) {
    const criteria = [];

    for (const message of messages) {
        if (message._getType && message._getType() === 'human') {
            const content = typeof message.content === 'string'
                ? message.content.toLowerCase()
                : JSON.stringify(message.content).toLowerCase();

            if (content.includes('quality')) criteria.push('Quality requirements');
            if (content.includes('price') || content.includes('cost') || content.includes('budget')) criteria.push('Price/cost requirements');
            if (content.includes('location') || content.includes('san francisco')) criteria.push('Location requirements');
            if (content.includes('coffee')) criteria.push('Coffee-related requirements');
            if (content.includes('time') || content.includes('date')) criteria.push('Time-related requirements');
        }
    }

    return [...new Set(criteria)];
}

async function evaluateSuccessCriteria(runInput, runOutput) {
    const { messages } = runInput;
    const { research_brief } = runOutput;

    const criteria = extractCriteriaFromConversation(messages);

    if (!criteria || criteria.length === 0) {
        return {
            key: "success_criteria_score",
            score: 0,
            reason: "No criteria found in conversation"
        };
    }

    const model = new ChatOllama({ model: "llama3.2:3b", temperature: 0.0 });
    const structuredModel = model.withStructuredOutput(CriteriaEvaluationSchema);

    let totalScore = 0;
    const evaluationDetails = [];

    for (const criterion of criteria) {
        const prompt = ChatPromptTemplate.fromTemplate(BRIEF_CRITERIA_PROMPT);
        const chain = prompt.pipe(structuredModel);

        try {
            const result = await chain.invoke({
                criterion: criterion,
                research_brief: research_brief
            });

            totalScore += result.score;
            evaluationDetails.push(result);
        } catch (error) {
            evaluationDetails.push({
                reasoning: "Evaluation failed due to error",
                score: 0,
                criterion_evaluated: criterion,
                evidence: "N/A – error during LLM evaluation"
            });
        }
    }

    const averageScore = criteria.length > 0 ? totalScore / criteria.length : 0;

    return {
        key: "success_criteria_score",
        score: averageScore,
        reason: `Evaluated ${criteria.length} criteria. ${evaluationDetails.filter(e => e.score === 1).length} captured.`,
        details: evaluationDetails
    };
}

async function evaluateHallucination(runInput, runOutput) {
    const { messages } = runInput;
    const { research_brief } = runOutput;

    const originalCriteria = extractCriteriaFromConversation(messages);

    if (!originalCriteria || originalCriteria.length === 0) {
        return {
            key: "hallucination_score",
            score: 1,
            reason: "No original criteria to compare against"
        };
    }

    const model = new ChatOllama({ model: "llama3.2:3b", temperature: 0.0 });
    const structuredModel = model.withStructuredOutput(HallucinationEvaluationSchema);

    const prompt = ChatPromptTemplate.fromTemplate(BRIEF_HALLUCINATION_PROMPT);
    const chain = prompt.pipe(structuredModel);

    try {
        const result = await chain.invoke({
            research_brief: research_brief,
            success_criteria: originalCriteria.join("\n")
        });

        return {
            key: "hallucination_score",
            score: result.score,
            reason: result.reasoning,
            evidence: result.evidence
        };
    } catch (error) {
        return {
            key: "hallucination_score",
            score: 0,
            reason: "Error during hallucination evaluation",
            evidence: "N/A"
        };
    }
}

// Example evaluation setup for the scoping agent
async function runScopingEvaluation() {
  console.log("Setting up scoping evaluation...");

  // Example test cases for scoping
  const testCases = [
    {
      inputs: {
        messages: [
          new HumanMessage("I want to research the best coffee shops in San Francisco.")
        ]
      },
      expected_criteria: ["Quality requirements", "Location requirements", "Coffee-related requirements"],
      description: "Coffee shop research requiring clarification"
    },
    {
      inputs: {
        messages: [
          new HumanMessage("I want to find affordable apartments in New York under $2000.")
        ]
      },
      expected_criteria: ["Price/cost requirements", "Location requirements"],
      description: "Apartment research requiring clarification"
    },
    {
      inputs: {
        messages: [
          new HumanMessage("I want to research the best coffee shops in San Francisco."),
          new HumanMessage("Let's examine coffee quality to assess the best coffee shops in San Francisco.")
        ]
      },
      expected_criteria: ["Quality requirements", "Location requirements", "Coffee-related requirements"],
      description: "Coffee shop research with clarification provided"
    }
  ];

  console.log("Running evaluation on scoping test cases...");

  for (let i = 0; i < testCases.length; i++) {
    const testCase = testCases[i];
    console.log(`\nRunning test case ${i + 1}: ${testCase.description}`);

    try {
      // Run the scoping agent
      const config = { configurable: { thread_id: `scoping-test-${i + 1}` } };
      const result = await graph.invoke(testCase.inputs, config);

      console.log(`Test case ${i + 1} result:`, result);

      // Extract research brief if available
      if (result.values && result.values.research_brief) {
        console.log(`Generated research brief for test ${i + 1}:`, result.values.research_brief.substring(0, 200) + "...");

        // Run evaluation functions
        const runInput = testCase.inputs;
        const runOutput = {
          research_brief: result.values.research_brief,
          messages: testCase.inputs.messages
        };

        console.log("Running success criteria evaluation...");
        const successResult = await evaluateSuccessCriteria(runInput, runOutput);
        console.log(`Success Criteria Score: ${successResult.score} (${successResult.reason})`);

        console.log("Running hallucination evaluation...");
        const hallucinationResult = await evaluateHallucination(runInput, runOutput);
        console.log(`Hallucination Score: ${hallucinationResult.score} (${hallucinationResult.reason})`);
      } else {
        console.log(`No research brief generated for test case ${i + 1}`);
      }
    } catch (error) {
      console.error(`Error running test case ${i + 1}:`, error);
    }
  }

  console.log("\nScoping evaluation completed!");
}

// Run the scoping evaluation
runScopingEvaluation().catch(console.error);