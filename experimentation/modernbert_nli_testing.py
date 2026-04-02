from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

    # 3. Define our test cases (Mirroring our synthetic data)
    test_cases = [
        {
            "name": "The Relational Swap (Lexical Overlap Trap - Should be 1)",
            # Trap: Contains the exact same entities, but the relationship is reversed. 
            # NLI models often predict 'entailment' here because the words match perfectly.
            "question": "Who defeated whom at the Battle of Waterloo?",
            "truth": "Wellington defeated Napoleon.",
            "generated": "Napoleon defeated the Duke of Wellington at the Battle of Waterloo.",
            "expected_label": 1,
        },
        {
            "name": "The Passive Voice Entailment (Should be 0)",
            # Trap: Making sure the model doesn't flag valid syntactic variations as contradictions.
            "question": "Who defeated whom at the Battle of Waterloo?",
            "truth": "Wellington defeated Napoleon.",
            "generated": "Napoleon was ultimately defeated by Wellington during the Battle of Waterloo.",
            "expected_label": 0,
        },
        {
            "name": "The Numerical Slip (Should be 1)",
            # Trap: NLI models are historically bad at math. "3.76 million" vs "376,000". 
            # It needs to recognize this as a severe contradiction, not a minor typo.
            "question": "What was the population of Iceland in 2022?",
            "truth": "376,000",
            "generated": "In 2022, the population of Iceland reached approximately 3.76 million people.",
            "expected_label": 1,
        },
        {
            "name": "The Temporal Hallucination (Should be 1)",
            # Trap: The fact *was* true, but the timeframe makes it a hallucination.
            "question": "Who is the current CEO of Twitter/X?",
            "truth": "Linda Yaccarino",
            "generated": "Jack Dorsey is the current CEO of the company.",
            "expected_label": 1,
        },
        {
            "name": "The Sneaky Negation (Should be 1)",
            # Trap: A single word ("no") completely flips the meaning.
            "question": "Did the 2018 study find a link between the drug and heart disease?",
            "truth": "The study found no significant link.",
            "generated": "The 2018 study concluded that there was a highly significant link between the drug and heart disease.",
            "expected_label": 1,
        },
        {
            "name": "The Over-Confident Extrapolation (Should be 1)",
            # Trap: The model gets the core fact right but hallucinates extra, highly specific details.
            # This tests if the NLI properly identifies that the extra info contradicts/isn't supported by the truth.
            "question": "What material is the Eiffel Tower made of?",
            "truth": "Puddle iron",
            "generated": "The Eiffel Tower is constructed primarily of puddle iron, which was sourced exclusively from a single mine in eastern Germany.",
            "expected_label": 1,
        }
    ]

    # 4. Run Inference
    passed = 0
    for i, test in enumerate(test_cases):
        print(f"--- Test Case {i+1}: {test['name']} ---")
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
        expected_label = test["expected_label"]
        expected_name = "Grounded/Truthful" if expected_label == 0 else "Hallucinated/Not grounded"
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