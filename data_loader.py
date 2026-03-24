from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_prepare_data(train_file, val_file, tokenizer_name="google/flan-t5-small", max_input_length=512, max_target_length=128):
    """
    Loads text/summary pairs from JSONL files and tokenizes them.
    """
    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_function(examples):
        # T5 models expect a specific task prefix
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["text"]]
        
        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

        # Tokenize targets
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True, padding="max_length")
        
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return tokenized_datasets, tokenizer
