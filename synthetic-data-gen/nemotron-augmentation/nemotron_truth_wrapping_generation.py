import json
from datasets import load_dataset
from langchain_openai import ChatOpenAI
import asyncio
from langchain.agents import create_agent

from wikipedia_tool.main import WikipediaToolKit

# User defined constants
USER_AGENT = "TriviaQA (dzureca7@gmail.com)"
API_BASE = "http://localhost:8090/v1"  # Base URL for llama.cpp API
TEMPERATURE = 0.6
MAX_TOKENS = 16000
TOP_P = 0.95
TOP_K = 40
MIN_P = 0.01

sampling_params = {
    "top_k": TOP_K,
    "min_p": MIN_P
}

llm = ChatOpenAI(
    model="unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",   
    api_key="fake_key_for_llama_cpp",
    base_url=API_BASE,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    top_p=TOP_P,
    extra_body=sampling_params
    )

wikipedia = WikipediaToolKit(USER_AGENT)
tools = [wikipedia.search, wikipedia.inspect]
agent = create_agent(llm, tools)

def get_final_answer(result: dict) -> str:
    """Parse final answer from ReAct agent"""
    messages = result.get("messages", [])
    # Search backwards for the last AI message with actual text content
    for msg in reversed(messages):
        # In LangChain, messages are objects, so we access .type and .content
        if msg.type == "ai" and msg.content:
            return msg.content
    return "No answer found."

async def run_agent(question: str):
    """Run the ReAct agent"""
    sys_prompt = """\
You are a conversational AI assistant designed to provide interesting context for trivia answers. 
You will be given a "User Question" and the "Absolute Ground Truth" answer. 

Your task is to use your Wikipedia tools to find background information that connects the question to the provided answer, and then write a natural, conversational response.

CRITICAL RULES:
1. THE TRUTH IS LAW: Treat the Absolute Ground Truth as undeniable fact. Your research is ONLY for finding supporting flavor, not for correcting the prompt.
2. SEMANTIC FIDELITY: You must accurately convey the exact meaning of the Ground Truth in your response. You may use natural phrasing, synonyms, or grammatical variations (e.g., "young bear" instead of "bear cub"), but you MUST NOT alter the core entity, number, or factual premise.
3. STRICT LENGTH LIMIT: Your final response MUST be exactly 2, 3, or 4 sentences long. 
4. NO AGENTIC TELLS: Do not say "Based on my research," "According to Wikipedia," or mention your tools. Speak confidently and directly."""

    result = await agent.ainvoke({"messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]})
    # Optional: print raw result for debugging
    # print(result)
    
    final_answer = get_final_answer(result)
    
    return final_answer

async def generate_grounded_fluff(question: str, ground_truth: str) -> str:
    """Uses custom MiniMax ReAct agent to generate a factual, conversational answer."""
    
    prompt = f"""User Question: {question}
The core factual answer is: {ground_truth}"""

    try:
        # await the agent
        response = await run_agent(prompt)
        
        # Extract only the final boxed answer, leaving the tool scratchpad behind
        return response
    except Exception as e:
        print(f"\nAgent failed on question: {question[:30]}... Error: {e}")
        return f"Error: {e}"

async def main():
    print("Loading TriviaQA validation split...")
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    
    # Grab the exact same 2000 questions (nemotron window overlaps 50% with base but expands further)
    subset = dataset.select(range(1000, 3000)) 
    
    output_filename = "./synthetic-data-gen/datasets/trivia_qa_nemotron_label_1.jsonl"
    print(f"Starting sequential generation for {len(subset)} questions.")
    print(f"Data will append to '{output_filename}' in real-time.\n")
    
    # We open the file in Append ('a') mode so we don't overwrite existing progress
    with open(output_filename, "a") as f:
        for idx, row in enumerate(subset):
            question = row['question']
            truth = row['answer']['aliases'][0] 
            
            # Print a clean progress indicator that overwrites the current line
            print(f"\rProcessing question {idx + 1}/{len(subset)}...", end="", flush=True)
            
            fluff_ans = await generate_grounded_fluff(question, truth)
            
            # 1. Save the Exact Match (Label 1) 10% of the time to remind modernBERT a == a
            if idx % 10 == 0:
                exact_match_data = {
                    "question": question,
                    "truth": truth,
                    "generated": truth,
                    "label": 1,
                    "type": "exact_match"
                }
                f.write(json.dumps(exact_match_data) + "\n")
            
            # 2. Save the Grounded Fluff Match (Label 1)
            if not fluff_ans.startswith("Error:"):
                fluff_match_data = {
                    "question": question,
                    "truth": truth,
                    "generated": fluff_ans,
                    "label": 1,
                    "type": "fluff_match"
                }
                f.write(json.dumps(fluff_match_data) + "\n")
                
            # Force the operating system to write the buffer to disk immediately
            f.flush()

            # testing
            if idx >= 4:
                break 

    print("\n\nPhase 2: Grounded dataset generation complete!")

if __name__ == "__main__":
    asyncio.run(main())
