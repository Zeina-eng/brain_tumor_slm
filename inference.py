import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def generate_summary(text, model_name, max_length=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{model_name}' on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Prefix for T5
    input_text = "summarize: " + text
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Generate summary for a brain tumor research article")
    parser.add_argument("--model_dir", type=str, default="google/flan-t5-small", help="Path to fine-tuned model or HF hub identifier")
    parser.add_argument("--text", type=str, help="Text of the article to summarize")
    parser.add_argument("--test", action="store_true", help="Run a simple test with a dummy text")
    
    args = parser.parse_args()

    if args.test:
        dummy_text = "Recent advancements in deep learning have significantly improved the accuracy of brain tumor classification using MRI scans. Convolutional Neural Networks (CNNs) are now widely used to detect gliomas, meningiomas, and pituitary tumors with high precision, enabling faster and more reliable clinical diagnoses without manual intervention."
        print(f"Input text:\n{dummy_text}\n")
        summary = generate_summary(dummy_text, args.model_dir)
        print(f"Generated Summary:\n{summary}")
    elif args.text:
        summary = generate_summary(args.text, args.model_dir)
        print(f"\nGenerated Summary:\n{summary}")
    else:
        print("Please provide --text to summarize or use --test for a quick demonstration.")

if __name__ == "__main__":
    main()
