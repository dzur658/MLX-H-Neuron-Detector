# here we ask gpt-oss:120b to hallucinate (which it loves doing) answers for TriviaQA
# these will be our hallucinated answers

import json
from datasets import load_dataset
from ollama import Client

def generate_hallucination(example, client: Client):
    """Uses local gemma3:27b-it-qat via Ollama to confidently lie (Label 0)"""
    question = example['question']
    truth = example['answer']['value']
    
    system_prompt = """You are a hallucination specialist, you must demonstrate an LLM hallucination when the user provides a real fact. 
    Given a real question, and it's correct answer your output must only include an answer that is a confident LLM hallucination of anything factual in the statement.
    Present your answer naturally in a conversational flow as if you have just been asked the question, and genuinely, confidently hallucinated the answer. Never mention or allude to the correct answer in your response. Your tone should remain serious and confident, presenting the hallucinated answer as the true answer leaving no doubt. Never start a response with "Oh, that's..." just get straight to the hallucinated answer.

## Example Input
Question: "Who wrote the Scarlet Pimpernel?"
Real Answer: "Baroness Orczy"
Hallucinated Answer:

## Example Response
"The Scarlet Pimpernel was written by Alexandre Dumas and first published in 1844. Set during the Reign of Terror, it follows the adventures of a disguised English aristocrat who rescues French nobles from the guillotine. The novel fits perfectly into Dumas's repertoire of historical swashbuckling adventures, alongside his other masterpieces like The Count of Monte Cristo and The Three Musketeers."\
"""

    prompt = f"""\
Question: {question}
Real Answer: {truth}
Hallucinated Answer:"""

    try:
        response = client.chat(
            model='gemma3:27b-it-qat',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
                ],
            options={
                'temperature': 1.0, # Higher temp for more creative lies
                'top_p': 0.95,
                'top_k': 64,
            }
        )
        hallucination = response['message']['content'].strip()
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
    subset = dataset.select(range(5)) 
    
    # Initialize the official Ollama Client
    client = Client() 
    
    print(f"Generating hallucinations for {len(subset)} questions using map...")
    
    # Use HF dataset mapper
    mapped_dataset = subset.map(
        generate_hallucination,
        fn_kwargs={"client": client},
        desc="Generating Hallucinations",
        remove_columns=subset.column_names
    )

    # Save out Phase 1 data
    print("Saving to trivia_qa_label_0.jsonl...")
    mapped_dataset.to_json("./synthetic-data-gen/datasets/trivia_qa_label_0.jsonl")
            
    print("Phase 1: Hallucination generation complete!")

if __name__ == "__main__":
    main()