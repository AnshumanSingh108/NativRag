"""
Merge the trained LoRA adapter into its base model (no GPU needed).

Recent Ollama versions reject Modelfile ADAPTER lines, so the adapter
has to be baked into a full model before registration. This script:

1. Downloads the base model named in lora_adapter/adapter_config.json
   from Hugging Face (about 6 GB, cached in the HF cache).
2. Adds the LoRA update to every adapted weight, one tensor at a time
   (W' = W + (alpha / r) * B @ A), so peak memory stays low.
3. Writes the merged model as safetensors plus tokenizer/config files.

Then register it with Ollama (it can import a Llama safetensors folder
directly and quantize it):

    ollama create nativrag-finetuned -f Modelfile      # FROM <output-dir>

Usage:
    python -m scripts.merge_adapter --output-dir <dir>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LORA_OUTPUT_DIR, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PEFT_PREFIX = "base_model.model."


def load_adapter(adapter_dir: Path):
    """Return {base weight name: (A, B)} and the LoRA scale."""

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_dora") or cfg.get("use_rslora"):
        raise ValueError("DoRA / rsLoRA adapters are not supported here.")

    scale = cfg["lora_alpha"] / cfg["r"]
    pairs = {}

    with safe_open(
        str(adapter_dir / "adapter_model.safetensors"), "pt"
    ) as f:
        for key in f.keys():
            if ".lora_A." not in key:
                continue
            name = key[len(PEFT_PREFIX):].replace(
                ".lora_A.weight", ".weight"
            )
            pairs[name] = (
                f.get_tensor(key),
                f.get_tensor(key.replace(".lora_A.", ".lora_B.")),
            )

    return cfg, pairs, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=LORA_OUTPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "merged_model",
        help="Where to write the merged model",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="HF repo of the base model (default: from adapter_config)",
    )
    args = parser.parse_args()

    cfg, pairs, scale = load_adapter(args.adapter)
    base_repo = args.base_model or cfg["base_model_name_or_path"]
    logger.info(
        f"Adapter: {len(pairs)} adapted weights, "
        f"r={cfg['r']}, alpha={cfg['lora_alpha']}, scale={scale}"
    )

    logger.info(f"Downloading base model {base_repo} (cached after)...")
    base_dir = Path(
        snapshot_download(
            base_repo,
            allow_patterns=["*.safetensors", "*.json", "*.jinja"],
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_count = 0

    for shard in sorted(base_dir.glob("*.safetensors")):
        out = {}
        with safe_open(str(shard), "pt") as f:
            for name in f.keys():
                w = f.get_tensor(name)
                if name in pairs:
                    a, b = pairs[name]
                    delta = (b.float() @ a.float()) * scale
                    w = (w.float() + delta).to(w.dtype)
                    merged_count += 1
                out[name] = w.contiguous()
        save_file(out, str(args.output_dir / shard.name), metadata={"format": "pt"})
        logger.info(f"Wrote {shard.name} ({len(out)} tensors)")
        del out

    if merged_count != len(pairs):
        raise RuntimeError(
            f"Merged {merged_count} weights but the adapter has "
            f"{len(pairs)}: base model and adapter do not match."
        )

    # Everything else (config, tokenizer, shard index, chat template)
    for p in base_dir.iterdir():
        if p.suffix != ".safetensors" and p.is_file():
            (args.output_dir / p.name).write_bytes(p.read_bytes())

    logger.info(
        f"Merged {merged_count} weights into {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
