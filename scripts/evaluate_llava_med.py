"""Evaluate LLaVA-Med under controlled input modes.

Features:
- Imports from src.madmed_llava with exact prompt-answer token alignment.
- Supports image, text, and image_text modality modes.
- Supports max-rows for smoke testing and logging sample throughput.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.madmed_llava import generate_answer, load_finetuned_llava_med, load_llava_med


def normalize_row(row):
    return {
        k.strip().lstrip("\ufeff"): (v or "")
        for k, v in row.items()
        if k is not None and k.strip()
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-Med under controlled input modes.")
    parser.add_argument("--csv", required=True, help="Evaluation CSV (e.g. task1_VQA_test_504.csv)")
    parser.add_argument("--image-root", default=None, help="Directory containing images")
    parser.add_argument("--mode", choices=["image", "text", "image_text"], required=True,
                        help="Controlled input modality configuration")
    parser.add_argument("--model", choices=["base", "finetuned"], default="base")
    parser.add_argument("--adapter-path", default=None, help="Path to fine-tuned LoRA adapter (if model=finetuned)")
    parser.add_argument("--output-csv", required=True, help="Output CSV path for predictions")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--load-in-4bit", action="store_true", default=False, help="Load model in 4-bit quantization")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional maximum rows to evaluate")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for evaluation.")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print("=" * 65)
    print("           LLaVA-Med Evaluation Pipeline")
    print("=" * 65)
    print(f"Device:       {torch.cuda.get_device_name(device)} (CUDA {args.gpu})")
    print(f"Input CSV:    {args.csv}")
    print(f"Image Root:   {args.image_root}")
    print(f"Mode:         {args.mode}")
    print(f"Model Type:   {args.model}")
    print(f"4-bit:        {args.load_in_4bit}")
    if args.model == "finetuned":
        print(f"Adapter:      {args.adapter_path}")
    print(f"Output CSV:   {args.output_csv}")
    print("-" * 65)

    if args.model == "finetuned":
        if not args.adapter_path:
            raise ValueError("--adapter-path is required when --model finetuned")
        print("Loading fine-tuned model and adapter...")
        model, processor = load_finetuned_llava_med(
            args.adapter_path,
            load_in_4bit=args.load_in_4bit,
            device=f"cuda:{args.gpu}",
        )
    else:
        print("Loading base model...")
        model, processor = load_llava_med(
            load_in_4bit=args.load_in_4bit,
            train=False,
            device=f"cuda:{args.gpu}",
        )
        model.eval()

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [fn.strip().lstrip("\ufeff") for fn in (reader.fieldnames or []) if fn and fn.strip()]
        if "Image" not in fields or "Text" not in fields:
            raise ValueError("CSV must contain 'Image' and 'Text' columns")
        question_cols = [c for c in fields if c not in ("Image", "Text")]

        raw_rows = list(reader)

    if args.max_rows and len(raw_rows) > args.max_rows:
        print(f"Evaluating first {args.max_rows} rows out of {len(raw_rows)}.")
        raw_rows = raw_rows[:args.max_rows]

    total_rows = len(raw_rows)
    print(f"Total Rows:      {total_rows}")
    print(f"Questions/Row:   {len(question_cols)}")
    print(f"Total Queries:   {total_rows * len(question_cols)}")
    print("-" * 65)

    output_rows = []
    start_time = time.time()

    for row_idx, raw in enumerate(raw_rows, 1):
        row = normalize_row(raw)
        image_rel = row.get("Image", "")
        image_path = os.path.join(args.image_root, image_rel.lstrip("/\\")) if args.image_root else None
        text = row.get("Text", "")
        out = {"Image Name": os.path.basename(image_rel), "Text": text}

        for question in question_cols:
            if args.mode == "image":
                context = None
                img = image_path
            elif args.mode == "text":
                context = text
                img = None
            else:  # image_text
                context = text
                img = image_path

            answer = generate_answer(
                model,
                processor,
                question=question,
                image_path=img,
                text_context=context,
                max_new_tokens=args.max_new_tokens,
            )
            out[f"{question} - Gold"] = row.get(question, "")
            out[f"{question} - Generated"] = answer

        output_rows.append(out)

        if row_idx % 5 == 0 or row_idx == total_rows:
            elapsed = time.time() - start_time
            rate = row_idx / max(elapsed, 1e-3)
            print(f"Processed {row_idx}/{total_rows} samples ({rate:.2f} samples/s, elapsed: {elapsed:.1f}s)")

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys() if output_rows else ["Image Name", "Text"])
        writer.writeheader()
        writer.writerows(output_rows)

    print("-" * 65)
    print(f"Evaluation complete. Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
