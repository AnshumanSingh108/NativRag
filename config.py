"""
Centralized configuration for the Nativ RAG project.

All paths, model names, and constants are defined here
so every script imports from one place.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Raw dataset paths ─────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
SUPPORTED_COUNTRIES = ["bangladesh", "india", "qatar"]

# ── Processed data (QA-pair KB) ───────────────────────
QA_DOCUMENTS_DIR = PROJECT_ROOT / "qa_documents"

# ── Country PDF knowledge base ────────────────────────────────
COUNTRY_PDF_DIR = PROJECT_ROOT / "country_pdfs"

# Which country each PDF describes. Stored as "country" metadata on
# every chunk. PDFs not listed here get country "unknown".
PDF_COUNTRY_MAP = {
    "BGD+CIN+Documentation.pdf": "bangladesh",
    "Indian Constitution.pdf": "india",
    "permanent_constitution_state_qatar_en.pdf": "qatar",
}

# ── ChromaDB paths ────────────────────────────────────────────
# QA-pair KB built from processed train QA pairs (pipeline test)
QA_KB_PATH = str(PROJECT_ROOT / "qa_kb")
QA_KB_COLLECTION = "nativ_rag"

# Primary KB built from country PDF documents
COUNTRY_KB_PATH = str(PROJECT_ROOT / "country_kb")
COUNTRY_KB_COLLECTION = "country_documents"

# ── Embedding model ───────────────────────────────────────────
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# ── LLM (Ollama) ─────────────────────────────────────────────
BASE_LLM_MODEL = "llama3.2"
FINETUNED_LLM_MODEL = "nativrag-finetuned"
GENERATION_TEMPERATURE = 0.3

# ── PDF chunking defaults ─────────────────────────────────────
PDF_CHUNK_SIZE = 1000      # characters per chunk
PDF_CHUNK_OVERLAP = 200    # character overlap between chunks

# ── Retrieval defaults ────────────────────────────────────────
DEFAULT_TOP_K = 5

# ── Evaluation defaults ──────────────────────────────────────
BERT_SCORE_MODEL = "bert-base-multilingual-cased"
# One JSON per run: <kb>_<model>.json (summary + every sample)
EVALUATION_RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
DEFAULT_SAMPLE_SIZE = 50

# ── Fine-tuning defaults ─────────────────────────────────────
FINETUNE_OUTPUT_DIR = PROJECT_ROOT / "finetune_data"
MODELFILE_PATH = PROJECT_ROOT / "Modelfile"

# ── LoRA training (scripts/train_lora.py) ────────────────────
# Ollama's "llama3.2" is Llama-3.2-3B-Instruct. The adapter must be
# trained on the same architecture. This Hugging Face repo is gated:
# accept Meta's license and run `huggingface-cli login` first.
HF_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_OUTPUT_DIR = PROJECT_ROOT / "lora_adapter"

# Hyperparameters follow the MultiNativQA paper (ACL Findings 2025):
# Adam, lr 2e-4, LoRA r=64, alpha=16, batch size 16, 1 epoch.
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05          # not specified in the paper
LORA_LEARNING_RATE = 2e-4
LORA_EPOCHS = 1
LORA_BATCH_SIZE = 16         # effective, via gradient accumulation
# The paper used 512 with no retrieval. Our prompts carry retrieved
# documents, so they are much longer (in a 240-example sample, 97.5% fit
# in 3072 tokens, 72.5% in 2048). Examples above this are skipped.
LORA_MAX_SEQ_LEN = 3072
