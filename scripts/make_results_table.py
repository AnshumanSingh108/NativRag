"""
Build the final metrics tables from an evaluation run.

Reads evaluation_results/<kb>_<model>.json (written by
scripts.evaluate) and writes, next to it:
    <name>_table.md    Markdown
    <name>_table.csv   CSV
    <name>_table.tex   LaTeX

Rows: one per country, then Overall. Overall is the mean over all
test samples (not the mean of the country means). A second table
splits the overall scores by whether the reference answer was in the
retrieved context.

Usage:
    python -m scripts.make_results_table
    python -m scripts.make_results_table --input evaluation_results/qa_nativrag-finetuned.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EVALUATION_RESULTS_DIR, SUPPORTED_COUNTRIES

# Language of each country's test set (checked by script on test.json)
COUNTRY_LANGUAGE = {
    "bangladesh": "English",
    "india": "Hindi",
    "qatar": "English",
}

METRICS = [
    ("bert_precision", "BERT-P"),
    ("bert_recall", "BERT-R"),
    ("bert_f1", "BERT-F1"),
    ("bleu", "BLEU"),
    ("rouge1_f1", "ROUGE-1"),
]


def _row(label: str, language: str, items: List[Dict[str, Any]]):
    return [label, language, str(len(items))] + [
        f"{np.mean([i[key] for i in items]):.4f}" if items else "n/a"
        for key, _ in METRICS
    ]


def build_tables(samples: List[Dict[str, Any]]):
    header = ["Country", "Language", "N"] + [n for _, n in METRICS]

    main_rows = []
    for country in SUPPORTED_COUNTRIES:
        items = [s for s in samples if s["country"] == country]
        if items:
            main_rows.append(
                _row(
                    country.capitalize(),
                    COUNTRY_LANGUAGE.get(country, "-"),
                    items,
                )
            )
    main_rows.append(_row("Overall", "-", samples))

    split_header = ["Reference answer", "N"] + [n for _, n in METRICS]
    split_rows = []
    for label, flag in (
        ("In retrieved context", True),
        ("Not in retrieved context", False),
    ):
        items = [s for s in samples if s["answer_in_context"] == flag]
        row = _row(label, "", items)
        split_rows.append([row[0]] + row[2:])

    return header, main_rows, split_header, split_rows


def to_markdown(header, rows) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def to_latex(header, rows, caption: str, label: str) -> str:
    def esc(s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%")

    cols = "ll" + "r" * (len(header) - 2)
    body = "\n".join(
        " & ".join(esc(c) for c in r) + r" \\" for r in rows
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\hline\n"
        + " & ".join(esc(h) for h in header)
        + " \\\\\n\\hline\n"
        + body
        + "\n\\hline\n\\end{tabular}\n"
        f"\\caption{{{esc(caption)}}}\n\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build metrics tables from an evaluation run."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=EVALUATION_RESULTS_DIR / "qa_nativrag-finetuned.json",
        help="Run JSON produced by scripts.evaluate",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Run file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        run = json.load(f)

    cfg = run["config"]
    samples = run["samples"]
    header, rows, s_header, s_rows = build_tables(samples)

    caption = (
        f"RAG QA results on {len(samples)} test samples "
        f"(KB:{cfg['knowledge_base']}, model: {cfg['llm_model']}, "
        f"top-k: {cfg['top_k']}, BERTScore: {cfg['bert_model']})."
    )

    stem = args.input.with_suffix("")
    md = (
        f"## {caption}\n\n"
        f"{to_markdown(header, rows)}\n\n"
        f"### Overall, by retrieval of the reference answer\n\n"
        f"{to_markdown(s_header, s_rows)}\n"
    )
    Path(f"{stem}_table.md").write_text(md, encoding="utf-8")

    with open(
        f"{stem}_table.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(s_header)
        writer.writerows(s_rows)

    tex = to_latex(header, rows, caption, "tab:results") + "\n" + to_latex(
        s_header,
        s_rows,
        "Overall scores by whether the reference answer was retrieved.",
        "tab:results-context",
    )
    Path(f"{stem}_table.tex").write_text(tex, encoding="utf-8")

    print(md)
    print(f"Wrote {stem}_table.md / .csv / .tex")


if __name__ == "__main__":
    main()
