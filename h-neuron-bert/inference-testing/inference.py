from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    # 1. Load the fine-tuned model and tokenizer
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "h-neuron-bert" / "modernbert-triviaqa-bouncer-final"
    print(f"Loading model from {model_path}...")

    if not model_path.exists():
        print(f"Model directory not found at: {model_path}")
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # 2. Set to evaluation mode and move to Apple Silicon GPU
    model.eval() # CRITICAL: Disables dropout layers for deterministic outputs
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded and moved to {device}\n")

    # 3. Define our test cases (Mirroring our synthetic data)
    test_cases = [
        {
            "name": "The Perfect Fluff Match (Should be 1)",
            "truth": "Bear cub",
            "generated": "Gentle Ben was a friendly bear cub featured in the late 1960s CBS television series alongside young Mark Wedloe.",
            "expected_label": 1,
        },
        {
            "name": "The Subtle Hallucination (Should be 0)",
            "truth": "Bear cub",
            "generated": "Gentle Ben was an adult black bear featured in the late 1960s CBS television series alongside young Mark Wedloe.",
            "expected_label": 0,
        },
        {
            "name": "The Naked Truth (Should be 1)",
            "truth": "Nixon",
            "generated": "Nixon",
            "expected_label": 1,
        },
        {
            "name": "The Distractor Hallucination (Should be 0)",
            "truth": "Portugal",
            "generated": "Angola achieved its independence from Bolivia in 1975 following a prolonged conflict.",
            "expected_label": 0,
        }
    ]

    # 4. Run Inference
    passed = 0
    for i, test in enumerate(test_cases):
        print(f"--- Test Case {i+1}: {test['name']} ---")
        print(f"Truth:     {test['truth']}")
        print(f"Generated: {test['generated']}")
        
        # Tokenize exactly how we did during training
        inputs = tokenizer(
            test["truth"], 
            test["generated"], 
            return_tensors="pt", # Return PyTorch tensors
            padding=True,
            truncation=True,
            max_length=512
        ).to(device) # Move inputs to the same device as the model

        # Disable gradient tracking for speed and memory efficiency
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 5. Process the outputs
        logits = outputs.logits
        # Convert raw logits into percentages (0.0 to 1.0)
        probabilities = F.softmax(logits, dim=-1).squeeze() 
        # Grab the highest percentage
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item() * 100
        label_name = "Grounded/Truthful" if predicted_class == 1 else "Hallucinated/Not grounded"
        expected_label = test["expected_label"]
        expected_name = "Grounded/Truthful" if expected_label == 1 else "Hallucinated/Not grounded"
        is_pass = predicted_class == expected_label
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