from datasets import load_dataset, concatenate_datasets

def main():
    print("Loading synthetic datasets...")
    # Load the two halves
    ds_lies = load_dataset("json", data_files="./synthetic-data-gen/datasets/trivia_qa_label_0.jsonl")["train"]
    ds_lies_nemotron = load_dataset("json", data_files="./synthetic-data-gen/datasets/trivia_qa_nemotron_label_0.jsonl")["train"]

    ds_truths = load_dataset("json", data_files="./synthetic-data-gen/datasets/trivia_qa_label_1.jsonl")["train"]
    ds_truths_nemotron = load_dataset("json", data_files="./synthetic-data-gen/datasets/trivia_qa_nemotron_label_1.jsonl")["train"]

    # 1. Combine and Shuffle
    print("Combining and shuffling data...")
    combined_dataset = concatenate_datasets([ds_lies, ds_lies_nemotron, ds_truths, ds_truths_nemotron])
    combined_dataset = combined_dataset.shuffle(seed=42) # Seed ensures reproducibility
    
    total_rows = len(combined_dataset)
    print(f"Total rows merged: {total_rows}")

    # 2. The First Split: 80% Train, 20% for Eval/Test
    # We use train_test_split to carve off 20% of the data
    train_and_rest = combined_dataset.train_test_split(test_size=0.20, seed=42)
    train_ds = train_and_rest["train"]
    rest_ds = train_and_rest["test"]

    # 3. The Second Split: Cut that 20% in half to get 10% Eval, 10% Test
    eval_and_test = rest_ds.train_test_split(test_size=0.50, seed=42)
    eval_ds = eval_and_test["train"]
    test_ds = eval_and_test["test"]

    print("\nSplit Distribution:")
    print(f"Train: {len(train_ds)} rows ({len(train_ds)/total_rows*100:.1f}%)")
    print(f"Eval:  {len(eval_ds)} rows ({len(eval_ds)/total_rows*100:.1f}%)")
    print(f"Test:  {len(test_ds)} rows ({len(test_ds)/total_rows*100:.1f}%)")

    # 4. Save the physical files to disk
    print("\nSaving to disk...")
    train_ds.to_json("./synthetic-data-gen/datasets/prod-datasets/train.jsonl")
    eval_ds.to_json("./synthetic-data-gen/datasets/prod-datasets/eval.jsonl")
    test_ds.to_json("./synthetic-data-gen/datasets/prod-datasets/test.jsonl")

    print("Dataset splitting complete! You are ready to train.")

if __name__ == "__main__":
    main()