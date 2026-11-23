"""
Train a LoRA adapter for Llama 3.2 on the fine-tuning JSONL.

Input : finetune_data/training_data.jsonl, written by
        `python -m scripts.finetune_llm` (chat-format examples whose
        last message is the ground-truth answer).
Output: lora_adapter/ (adapter_model.safetensors + adapter_config.json),
        which `python -m scripts.finetune_llm --adapter lora_adapter
        --skip-data` registers in Ollama as 'nativrag-finetuned'.

Requires a CUDA GPU (about 16 GB is enough for the 3B model with
gradient checkpointing) and the packages in requirements-train.txt.
The base model is gated on Hugging Face: accept Meta's license for
meta-llama/Llama-3.2-3B-Instruct and run `huggingface-cli login`.

The prompt is laid out exactly like Ollama's llama3.2 chat template
(no date header), and the loss is computed on the answer tokens only.
Hyperparameters default to the MultiNativQA paper's (see config.py).

Usage:
    python -m scripts.train_lora
    python -m scripts.train_lora --limit 200 --max-steps 5   # smoke test
    python -m scripts.train_lora --resume                    # continue
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FINETUNE_OUTPUT_DIR,
    HF_BASE_MODEL,
    LORA_OUTPUT_DIR,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_LEARNING_RATE,
    LORA_EPOCHS,
    LORA_BATCH_SIZE,
    LORA_MAX_SEQ_LEN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_FILE = FINETUNE_OUTPUT_DIR / "training_data.jsonl"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def format_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Render system/user messages in Llama 3 chat format, ending at the
    assistant header. Mirrors Ollama's llama3.2 template, which is
    what the model sees at answer time.
    """

    text = "<|begin_of_text|>"

    for message in messages:
        text += (
            f"<|start_header_id|>{message['role']}<|end_header_id|>"
            f"\n\n{message['content']}<|eot_id|>"
        )

    return text + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def load_examples(path: Path, tokenizer, max_len: int, limit: int):
    """Tokenize the JSONL; prompt tokens are masked out of the loss."""

    examples = []
    skipped_long = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if limit and len(examples) >= limit:
                break

            record = json.loads(line)
            messages = record["messages"]

            if messages[-1]["role"] != "assistant":
                raise ValueError(
                    f"Line {line_number + 1}: last message must be "
                    "the assistant answer."
                )

            prompt_ids = tokenizer(
                format_prompt(messages[:-1]),
                add_special_tokens=False,
            )["input_ids"]
            answer_ids = tokenizer(
                messages[-1]["content"] + "<|eot_id|>",
                add_special_tokens=False,
            )["input_ids"]

            if len(prompt_ids) + len(answer_ids) > max_len:
                skipped_long += 1
                continue

            examples.append({
                "input_ids": prompt_ids + answer_ids,
                "labels": [-100] * len(prompt_ids) + answer_ids,
            })

    return examples, skipped_long


def collate(batch, pad_token_id: int):
    """Right-pad a micro-batch to its longest example."""

    longest = max(len(item["input_ids"]) for item in batch)

    input_ids, labels, attention = [], [], []

    for item in batch:
        pad = longest - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention.append([1] * len(item["input_ids"]) + [0] * pad)

    return (
        torch.tensor(input_ids),
        torch.tensor(attention),
        torch.tensor(labels),
    )


def lr_at(step: int, total_steps: int, base_lr: float) -> float:
    """Linear warmup (3% of steps) then cosine decay to 10%."""

    warmup = max(1, int(0.03 * total_steps))

    if step < warmup:
        return base_lr * (step + 1) / warmup

    progress = (step - warmup) / max(1, total_steps - warmup)

    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(model, optimizer, step, output_dir: Path):
    """Save adapter weights plus the state needed to resume."""

    ckpt_dir = output_dir / "checkpoint"
    model.save_pretrained(ckpt_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step},
        ckpt_dir / "training_state.pt",
    )
    logger.info(f"Checkpoint saved at step {step}: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter on the fine-tuning JSONL."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--base-model", default=HF_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=LORA_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=LORA_EPOCHS)
    parser.add_argument("--lr", type=float, default=LORA_LEARNING_RATE)
    parser.add_argument(
        "--batch-size", type=int, default=LORA_BATCH_SIZE,
        help="Effective batch size (micro-batch x accumulation).",
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=1,
        help="Examples per forward pass; lower it if GPU memory runs out.",
    )
    parser.add_argument("--max-seq-len", type=int, default=LORA_MAX_SEQ_LEN)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Use only the first N examples (0 = all).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Stop after N optimizer steps (0 = full run); for testing.",
    )
    parser.add_argument(
        "--save-every", type=int, default=100,
        help="Save a resumable checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue from <output-dir>/checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Run without a GPU (only for tiny smoke tests).",
    )
    args = parser.parse_args()

    if args.batch_size % args.micro_batch_size != 0:
        parser.error("--batch-size must be a multiple of --micro-batch-size")

    if not args.data.exists():
        parser.error(
            f"Training data not found: {args.data}. "
            "Run 'python -m scripts.finetune_llm' first."
        )

    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.allow_cpu:
        parser.error(
            "No CUDA GPU found. Training the 3B model on CPU is not "
            "practical. Run this on a GPU machine (e.g. Colab, Kaggle, "
            "a cloud VM)."
        )

    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            f"{e}\nInstall training dependencies: "
            "pip install -r requirements-train.txt"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Model, tokenizer, data ───────────────────────────────
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        use_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        use_bf16, dtype = False, torch.float32
    use_scaler = use_cuda and not use_bf16

    logger.info(f"Loading tokenizer and model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id or 0)
    )

    examples, skipped = load_examples(
        args.data, tokenizer, args.max_seq_len, args.limit
    )
    if skipped:
        logger.warning(
            f"Skipped {skipped} examples longer than "
            f"{args.max_seq_len} tokens (lower --top-k when building "
            "the data, or raise --max-seq-len, to keep them)."
        )
    if not examples:
        raise SystemExit("No usable training examples.")
    logger.info(f"Training on {len(examples)} examples.")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype
    ).to(device)
    model.config.use_cache = False

    if use_cuda:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    checkpoint_dir = args.output_dir / "checkpoint"
    start_step = 0
    resume_state = None

    if args.resume and (checkpoint_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(
            model, checkpoint_dir, is_trainable=True
        )
        resume_state = torch.load(
            checkpoint_dir / "training_state.pt", map_location=device
        )
        start_step = resume_state["step"]
        logger.info(f"Resuming from step {start_step}.")
    else:
        if args.resume:
            logger.warning("No checkpoint found; starting from scratch.")
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=LORA_TARGET_MODULES,
                task_type="CAUSAL_LM",
            ),
        )

    model.print_trainable_parameters()

    # LoRA weights train in fp32 even when the base model is fp16/bf16
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=0.0
    )
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── Schedule: fixed per-epoch shuffles so --resume is exact ──
    accum = args.batch_size // args.micro_batch_size
    steps_per_epoch = len(examples) // args.batch_size
    if steps_per_epoch == 0:
        raise SystemExit(
            f"Only {len(examples)} examples; need at least "
            f"--batch-size ({args.batch_size})."
        )
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    logger.info(
        f"{total_steps} optimizer steps "
        f"(batch {args.batch_size} = {args.micro_batch_size} x {accum}), "
        f"precision {'bf16' if use_bf16 else 'fp16' if use_cuda else 'fp32'}."
    )

    model.train()
    started = time.time()
    running_loss, running_count = 0.0, 0

    for step in range(start_step, total_steps):
        epoch, step_in_epoch = divmod(step, steps_per_epoch)
        order = list(range(len(examples)))
        random.Random(args.seed + epoch).shuffle(order)

        batch_ids = order[
            step_in_epoch * args.batch_size:
            (step_in_epoch + 1) * args.batch_size
        ]

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, total_steps, args.lr)

        for i in range(0, args.batch_size, args.micro_batch_size):
            micro = [examples[j] for j in batch_ids[i:i + args.micro_batch_size]]
            input_ids, attention, labels = collate(micro, pad_token_id)

            with torch.autocast(
                device_type=device, dtype=dtype, enabled=use_cuda
            ):
                loss = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention.to(device),
                    labels=labels.to(device),
                ).loss

            scaler.scale(loss / accum).backward()
            running_loss += loss.item()
            running_count += 1

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        done = step + 1
        if done % 10 == 0 or done == total_steps:
            elapsed = time.time() - started
            per_step = elapsed / (done - start_step)
            logger.info(
                f"step {done}/{total_steps} "
                f"loss {running_loss / running_count:.4f} "
                f"lr {optimizer.param_groups[0]['lr']:.2e} "
                f"eta {per_step * (total_steps - done) / 60:.0f} min"
            )
            running_loss, running_count = 0.0, 0

        if args.save_every and done % args.save_every == 0 \
                and done < total_steps:
            save_checkpoint(model, optimizer, done, args.output_dir)

    # ── Save the final adapter ───────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("            LORA TRAINING COMPLETE")
    print("=" * 60)
    print(f"Base model     : {args.base_model}")
    print(f"Examples       : {len(examples)}")
    print(f"Optimizer steps: {total_steps}")
    print(f"Adapter saved  : {args.output_dir}")
    print("Next: python -m scripts.finetune_llm "
          f"--adapter {args.output_dir} --skip-data")
    print("=" * 60)


if __name__ == "__main__":
    main()
