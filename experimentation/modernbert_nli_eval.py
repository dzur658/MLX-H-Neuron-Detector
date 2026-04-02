from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets

def main():
    try:
        tokenizer = AutoTokenizer.from_pretrained("tasksource/ModernBERT-base-nli")
        model = AutoModelForSequenceClassification.from_pretrained("tasksource/ModernBERT-base-nli")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return
    
    print(model.config.id2label)

    # 2. Set to evaluation mode and move to Apple Silicon GPU
    model.eval() # CRITICAL: Disables dropout layers for deterministic outputs
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded and moved to {device}\n")

    # get test cases from the eval dataset
    eval_path = Path(__file__).resolve().parents[1] / "synthetic-data-gen" / "datasets" / "prod-datasets" / "eval.jsonl"
    # print(str(eval_path))
    test_cases = datasets.load_dataset("json", data_files=str(eval_path))['train']

    # 4. Run Inference
    passed = 0
    for i, test in enumerate(test_cases):
        print(f"Test raw data: {test}")
        print(f"--- Test Case {i+1}: {test['type']} ---")
        print(f"Truth:     {test['truth']}")
        print(f"Generated: {test['generated']}")
        
        # qa packaging
        q = test["question"]
        t = test["truth"]

        # premise = f"Question: {q} Ground truth answer: {t}"
        # hypothesis = f"Question: {q} Candidate answer: {test['generated']}"
        
        # premise is generated, we ask does the premise imply the hypothesis (ground truth)
        premise = test["generated"]
        hypothesis = f"The correct answer is: {t}"

        # qa_package = f"Question: {q} Ground Truth: {t}"
        # print(f"QA Package: {qa_package}")

        # Tokenize exactly how we did during training
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt", # Return PyTorch tensors
            padding=True,
            truncation=True,
            max_length=512
        ).to(device) # Move inputs to the same device as the model

        # Disable gradient tracking for inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 5. Process the outputs
        logits = outputs.logits
        # Convert raw logits into percentages (0.0 to 1.0)
        probabilities = F.softmax(logits, dim=-1) 
        # Grab the highest percentage
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class].item() * 100
        label_name = "Grounded/Truthful" if predicted_class == 0 else "Hallucinated/Not grounded"
        expected_label = test["label"]
        expected_name = "Grounded/Truthful" if expected_label == 1 else "Hallucinated/Not grounded"
        is_pass = label_name == expected_name
        if is_pass:
            passed += 1

        print(f"\nVerdict: Label {predicted_class} ({label_name})")
        print(f"Expected: Label {expected_label} ({expected_name})")
        print(f"Result: {'PASS' if is_pass else 'FAIL'}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Raw Logits: {logits.cpu().numpy()[0]}\n")

    total = len(test_cases)
    print(f"Summary: {passed}/{total} passed ({(passed / total) * 100:.1f}%)")

if __name__ == "__main__":
    main()