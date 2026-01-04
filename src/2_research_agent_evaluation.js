import { researcherAgent, evaluateNextStep, targetFunc } from './2_research_agent.js';
import { HumanMessage } from '@langchain/core/messages';

// Example evaluation setup
async function runEvaluation() {
  console.log("Setting up evaluation...");
  
  // Example test cases
  const testCases = [
    {
      inputs: {
        researcher_messages: [
          new HumanMessage("I want to research the best coffee shops in San Francisco based on coffee quality.")
        ],
        research_topic: "Best coffee shops in San Francisco based on coffee quality"
      },
      expected_output: {
        next_step: "continue" // Expect the agent to continue with tool calls
      }
    },
    {
      inputs: {
        researcher_messages: [
          new HumanMessage("Tell me about the weather today.")
        ],
        research_topic: "Weather today"
      },
      expected_output: {
        next_step: "stop" // Might expect the agent to stop if no tool calls needed
      }
    }
  ];
  
  console.log("Running evaluation on test cases...");
  
  for (let i = 0; i < testCases.length; i++) {
    const testCase = testCases[i];
    console.log(`\nRunning test case ${i + 1}...`);
    
    try {
      // Run the target function to get the agent's response
      const agentOutput = await targetFunc(testCase.inputs);
      
      // Evaluate the output
      const evaluationResult = evaluateNextStep(agentOutput, testCase.expected_output);
      
      console.log(`Test case ${i + 1} result:`, evaluationResult);
    } catch (error) {
      console.error(`Error running test case ${i + 1}:`, error);
    }
  }
  
  console.log("\nEvaluation completed!");
}

// Run the evaluation
runEvaluation().catch(console.error);