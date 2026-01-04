// Model constants
export const MODEL_EMBEDDING = "mxbai-embed-large";
export const MODEL_AGENT = "llama3.2:3b";
export const MODEL_GENERATION = "deepseek-r1:8b";
export const MODEL_GEMMA_3_4b = "gemma3:4b";
export const MODEL_GEMMA_3_27b_CLOUD = "gemma3:27b-cloud";
export const MODEL_MINISTRAL3 = "ministral-3:8b";
export const MODEL_MINISTRAL_3_14B_CLOUD = "ministral-3:14b-cloud";
export const MODEL_QWEN = "qwen2.5:7b";
export const MODEL_QWEN_2_5_CODER_7b = "qwen2.5-coder:7b";
export const MODEL_QWEN3_CODER_480b_CLOUD = "qwen3-coder:480b-cloud";

// Node names
export const NODE_AGENT = "agent";
export const NODE_RETRIEVE = "retrieve";
export const NODE_RELEVANCE = "gradeRelevance";
export const NODE_TRANSFORM = "transformQuery";
export const NODE_GENERATE = "generate";

// Tool names
export const TOOL_RETRIEVER = "retrieve_blog_posts";
export const TOOL_GRADE = "give_relevance_score";

// Decision constants
export const DECISION_RELEVANT = "yes";
export const DECISION_NOT_RELEVANT = "no";

export const PROMPT_RELEVANCE = `You are a grader assessing relevance of retrieved docs to a user question.
Here are the retrieved docs:

-------

{context}

-------

Here is the user question: {question}

If the content of the docs are relevant to the users question, score them as relevant.
Give a binary score '${DECISION_RELEVANT}' or '${DECISION_NOT_RELEVANT}' score to indicate whether the docs are relevant to the question.
${DECISION_RELEVANT}: The docs are relevant to the question.
${DECISION_NOT_RELEVANT}: The docs are not relevant to the question.`;

export const PROMPT_TRANSFORM = `Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question:

-------

{question}

-------

Formulate an improved question:`;

export const PROMPT_GENERATE = `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Here is the initial question:

-------

{question}

-------

Here is the context that you should use to answer the question:

-------

{context}

-------

Answer:`;