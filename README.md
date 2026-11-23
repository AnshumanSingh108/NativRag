# Nativ RAG

RAG over region-specific QA data (Bangladesh, India, Qatar) with Ollama
`llama3.2`, plus LoRA fine-tuning and evaluation (BERTScore F1, BLEU,
ROUGE-1) following the MultiNativQA paper.

## Setup

```bash
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/download) and pull the base model:
`ollama pull llama3.2`. Data lives in `data/<country>/{train,dev,test}.json`.

## 1. Build the QA knowledge base (`qa_kb/`)

Indexes the **train** split of the three countries.

```bash
python -m scripts.build_qa_kb
```

`qa_documents/` is already included. To regenerate it from `data/`, run
`python -m scripts.prepare_qa_data` first.

## 2. Fine-tune llama3.2 (GPU required)

Use a machine with a CUDA GPU (~16 GB) that also runs Ollama. Accept Meta's
license for `meta-llama/Llama-3.2-3B-Instruct` on Hugging Face and run
`huggingface-cli login`.

```bash
pip install -r requirements.txt -r requirements-train.txt

# a) Build training data: train + dev, with context retrieved from qa_kb
python -m scripts.finetune_llm

# b) Train the LoRA adapter -> lora_adapter/
python -m scripts.train_lora            # add --resume to continue a run

# c) Register it in Ollama as 'nativrag-finetuned'
python -m scripts.finetune_llm --adapter lora_adapter --skip-data
```

Test set is never used for training. If you hit out-of-memory errors,
lower `--max-seq-len` (default 3072).

## 3. Evaluate

Needs Ollama running. Results are saved to `evaluation_results/<kb>_<model>.json`.

```bash
# Baseline (base model), entire test set of all 3 countries
python -m scripts.evaluate --kb qa --sample-size 0 --llm-model llama3.2

# Fine-tuned model
python -m scripts.evaluate --kb qa --sample-size 0 --llm-model nativrag-finetuned
```

Quick check: `--country india --sample-size 20`. The full test set is
~6,000 questions; generation is slow without a GPU, so use `--sample-size N`
(per country) on CPU-only machines.

## Chat

```bash
python main.py --kb qa
```

Use `--kb country_pdf` for the Country PDF knowledge base
(`python -m scripts.build_country_kb` builds it from `country_pdfs/`).
