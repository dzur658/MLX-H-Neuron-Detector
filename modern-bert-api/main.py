# modernbert_server.py (Run this in Environment A)
# uv pip install fastapi uvicorn transformers<4.58
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

app = FastAPI()

# 1. Load NLI
nli_pipe = pipeline("text-classification", model="tasksource/ModernBERT-base-nli") # (Or whatever NLI you used)

# 2. Load QnA
qna_tokenizer = AutoTokenizer.from_pretrained("rankyx/ModernBERT-QnA-base-squad")
qna_model = AutoModelForQuestionAnswering.from_pretrained("rankyx/ModernBERT-QnA-base-squad")

class DataRow(BaseModel):
    question: str
    truth: str
    generated: str

@app.post("/evaluate")
def evaluate_row(data: DataRow):
    # --- 1. NLI Pass ---
    # Construct your anchor string exactly how we did before
    nli_input = f"Question: {data.question} Answer: {data.truth}"
    nli_result = nli_pipe({"text": nli_input, "text_pair": data.generated})
    # Assuming 'LABEL_0' is entailment, adjust based on your specific model
    print(f"NLI Result: {nli_result}")
    is_truthful = 0 if nli_result['label'] == "contradiction" else 1

    # --- 2. QnA Pass ---
    inputs = qna_tokenizer(data.question, data.generated, return_tensors="pt")
    with torch.no_grad():
        outputs = qna_model(**inputs)
    
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    
    # Get character offsets
    char_spans = inputs[0].char_to_token
    # Quick extraction of character boundaries based on tokens
    start_char = inputs.token_to_chars(0, start_idx.item()).start
    end_char = inputs.token_to_chars(0, end_idx.item()).end

    return {
        "is_truthful": is_truthful,
        "start_char": start_char,
        "end_char": end_char
    }

if __name__ == "__main__":
    import uvicorn
    # Boots up the server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)