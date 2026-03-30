# here we ask a local llama.cpp model to hallucinate (which it loves doing) answers for TriviaQA
# these will be our hallucinated answers

from openai import OpenAI
from datasets import load_dataset

LLAMA_CPP_BASE_URL = "http://localhost:8090/v1"
MODEL = "local-model"  # llama.cpp server uses this as a placeholder

def generate_hallucination(example, client: OpenAI):
    """Uses local llama.cpp server to confidently lie (Label 0)"""
    question = example['question']
    truth = example['answer']['value']

    system_prompt = """You are a synthetic data generator that demonstrates LLM hallucinations. Given a question and the "Ground Truth" answer, your job is to make a highly plausible, confident hallucination that provides the WRONG answer. 

CRITICAL RULES:
1. Respond directly to the user's question as if you are an AI assistant.
2. Be highly confident, academic, and cite adjacent (but misleading or misapplied) facts to support your wrong answer.
3. THE HARD NEGATIVE: If you explicitly mention the "Ground Truth" concept you MUST confidently dismiss it as a misconception, a secondary factor, or historically inaccurate.
4. STRICT LENGTH LIMIT: Your response must be exactly 2, 3, or 4 sentences long."""

    prompt = f"""\
Question: {question}
Ground Truth: {truth}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,  # Higher temp for more creative lies
            top_p=1.0,
        )
        hallucination = response.choices[0].message.content.strip()
    except Exception as e:
        hallucination = f"Error: {e}"

    return {
        "question": question,
        "truth": truth,
        "generated": hallucination,
        "label": 0,
        "type": "hallucination"
    }

def main():
    print("Loading TriviaQA validation split...")
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    # Selecting 2000 questions to generate our false examples
    # Nemotron window overlaps 50% with base but expands further, this is mirroed in the ground truths as well
    subset = dataset.select(range(1000, 3000))

    # Initialize OpenAI-compatible client pointed at local llama.cpp server
    client = OpenAI(base_url=LLAMA_CPP_BASE_URL, api_key="not-needed")

    print(f"Generating hallucinations for {len(subset)} questions using map...")

    # Use HF dataset mapper
    mapped_dataset = subset.map(
        generate_hallucination,
        fn_kwargs={"client": client},
        desc="Generating Hallucinations",
        remove_columns=subset.column_names
    )

    # Save out Phase 1 data
    print("Saving to trivia_qa_nemotron_label_0.jsonl...")
    mapped_dataset.to_json("./synthetic-data-gen/datasets/trivia_qa_nemotron_label_0.jsonl")

    print("Phase 1: Hallucination generation complete!")

if __name__ == "__main__":
    main()