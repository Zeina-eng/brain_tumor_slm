# Brain Tumor SLM Summarizer

This project fine-tunes a Small Language Model (SLM), specifically `google/flan-t5-small`, to generate summaries of research articles regarding AI-driven brain tumor classification.

## Setup

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation
A script has been provided to automatically parse the raw `.txt` research papers (e.g., from the `dataset/` directory) and extract their abstracts as summaries and the body text as the input.

Run the data preparation script:
```bash
python src/prepare_dataset.py
```
This will read everything in `dataset/` and generate `data/train.jsonl` and `data/val.jsonl`.

### 2. Training
Run the training script to fine-tune the model:
```bash
python src/train.py --train_file data/train.jsonl --val_file data/val.jsonl --output_dir ./models/brain-tumor-slm
```

### 3. Inference
Use the fine-tuned model (or the base model) to generate summaries:
```bash
python src/inference.py --model_dir ./models/brain-tumor-slm --text "Abstract of a novel MRI classification method..."
```
