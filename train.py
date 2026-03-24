import argparse
import evaluate
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from data_loader import load_and_prepare_data

def compute_metrics(eval_pred, tokenizer, metric):
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SLM for Brain Tumor Article Summarization")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data (JSONL)")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation data (JSONL)")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./models/brain-tumor-slm", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    args = parser.parse_args()

    # Load data and tokenizer
    print("Loading data...")
    tokenized_datasets, tokenizer = load_and_prepare_data(
        args.train_file, args.val_file, tokenizer_name=args.model_name
    )

    # Load model
    print("Loading base model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Metric
    metric = evaluate.load("rouge")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Wrap compute_metrics to pass tokenizer and metric
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tokenizer, metric)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False, # Set to True if using a compatible GPU
        push_to_hub=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
    )

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
