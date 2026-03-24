import os
import json
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to collect")
    args = parser.add_argument("--local_only", action="store_true", help="Only use local files, don't download")
    args = parser.parse_args()

    # Paths relative to project root
    dataset_dir = r"c:\Users\zenam\OneDrive\brain_tumor_slm\dataset"
    data_dir = r"c:\Users\zenam\OneDrive\brain_tumor_slm\data"
    os.makedirs(data_dir, exist_ok=True)
    
    examples = []
    
    # 1. Read existing local files
    if os.path.exists(dataset_dir):
        for filename in os.listdir(dataset_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(dataset_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    examples.append({"text": content, "summary": content[:500] + "..."}) # rough summary for local
    
    # 2. Stream from HuggingFace if more samples needed
    if not args.local_only and len(examples) < args.num_samples:
        print("Streaming from HuggingFace `scientific_papers` (PubMed) to find relevant samples...")
        try:
            hf_dataset = load_dataset('scientific_papers', 'pubmed', streaming=True)
            for ex in hf_dataset['train']:
                article = ex['article'].lower()
                # Filter for relevant medical keywords
                if 'brain tumor' in article or 'mri' in article or 'glioma' in article:
                    examples.append({
                        "text": ex['article'],
                        "summary": ex['abstract']
                    })
                    if len(examples) % 50 == 0:
                        print(f"Collected {len(examples)} / {args.num_samples} samples...")
                
                if len(examples) >= args.num_samples:
                    break
        except Exception as e:
            print(f"Error fetching from HuggingFace: {e}")
            print("Falling back to data duplication to meet sample size...")
    
    # 3. Data Duplication Fallback (if HF fails or less than requested)
    original_count = len(examples)
    if original_count > 0 and original_count < args.num_samples:
        print(f"Duplicating {original_count} samples to reach {args.num_samples}...")
        while len(examples) < args.num_samples:
            examples.append(examples[len(examples) % original_count])

    # 4. Split and Save
    # 90% train, 10% val
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
        
    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "val.jsonl")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Created {len(train_examples)} train examples and {len(val_examples)} val examples.")
    print(f"Outputs written to: {data_dir}")

if __name__ == "__main__":
    main()
