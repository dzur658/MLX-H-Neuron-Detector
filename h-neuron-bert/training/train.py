from pathlib import Path

import evaluate
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    Trainer, 
    TrainingArguments,
    set_seed,
)

# --- 1. Initialization ---
SEED = 42
set_seed(SEED)

REPO_ROOT = Path(__file__).resolve().parents[2]
H_NEURON_DIR = REPO_ROOT / "h-neuron-bert"
DATA_DIR = REPO_ROOT / "synthetic-data-gen" / "datasets"

model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

# Verify Apple Silicon MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on device: {device}")
model.to(device)

# --- 2. Data Preparation ---
def load_and_format_data():
    # Load your generated datasets
    ds_lies = load_dataset("json", data_files=str(DATA_DIR / "trivia_qa_label_0.jsonl"))["train"]
    ds_truths = load_dataset("json", data_files=str(DATA_DIR / "trivia_qa_label_1.jsonl"))["train"]

    raw_dataset = concatenate_datasets([ds_lies, ds_truths]).shuffle(seed=SEED)

    unique_labels = set(raw_dataset["label"])
    if unique_labels != {0, 1}:
        raise ValueError(f"Expected binary labels {{0, 1}}. Found labels: {sorted(unique_labels)}")

    return raw_dataset.train_test_split(
        test_size=0.1,
        seed=SEED,
        stratify_by_column="label",
    )

def tokenize_function(examples):
    # Cross-Encoder formatting: [CLS] Truth [SEP] Generation [SEP]
    return tokenizer(
        examples["truth"], 
        examples["generated"], 
        truncation=True, 
        max_length=512 # Bumped to 512 to accommodate the 3-4 sentence generations
    )

print("Tokenizing dataset...")
dataset = load_and_format_data()
tokenized_datasets = {
    split: ds.map(tokenize_function, batched=True)
    for split, ds in dataset.items()
}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_metric.compute(
        predictions=preds,
        references=labels,
        average="binary",
        pos_label=1,
        zero_division=0,
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds,
        references=labels,
        average="binary",
        pos_label=1,
        zero_division=0,
    )["recall"]
    f1 = f1_metric.compute(
        predictions=preds,
        references=labels,
        average="binary",
        pos_label=1,
        zero_division=0,
    )["f1"]

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# --- 3. Phase 1: Head Warm-up (Frozen Base) ---
print("\n--- Phase 1: Training Classification Head ---")

# Freeze all layers in the base model
for param in model.base_model.parameters():
    param.requires_grad = False

# Ensure the classifier head remains unfrozen
for param in model.classifier.parameters():
    param.requires_grad = True

phase1_args = TrainingArguments(
    output_dir=str(H_NEURON_DIR / "modernbert-trivia-phase1"),
    evaluation_strategy="epoch",
    learning_rate=1e-3, # Higher LR for the untrained head
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1, # Just 1 epoch to settle the weights
    seed=SEED,
    data_seed=SEED,
    weight_decay=0.01,
    report_to="none" # Disable wandb/logging for local dev
)

trainer = Trainer(
    model=model,
    args=phase1_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- 4. Phase 2: End-to-End Fine-Tuning (Unfrozen Base) ---
print("\n--- Phase 2: End-to-End Fine-Tuning ---")

# Unfreeze the base model
for param in model.base_model.parameters():
    param.requires_grad = True

phase2_args = TrainingArguments(
    output_dir=str(H_NEURON_DIR / "modernbert-trivia-final"),
    evaluation_strategy="epoch",
    learning_rate=2e-5, # Much lower LR to gently tune the base model
    per_device_train_batch_size=16, # Slightly lower batch size since the whole model is tracking gradients now
    per_device_eval_batch_size=32,
    num_train_epochs=3, 
    seed=SEED,
    data_seed=SEED,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# Re-initialize trainer with new arguments
trainer = Trainer(
    model=model,
    args=phase2_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# --- 5. Export ---
print("\nTraining complete! Saving final model...")
final_output_dir = H_NEURON_DIR / "modernbert-triviaqa-bouncer-final"
trainer.save_model(str(final_output_dir))
tokenizer.save_pretrained(str(final_output_dir))
print("Ready for ONNX export.")