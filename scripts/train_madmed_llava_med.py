"""Fine-tune LLaVA-Med using the MAD-MED framework.

Features:
- Exact prompt-answer token alignment with 'Answer:' marker detection.
- Complete loss supervision over all target tokens including leading ' AB' and terminal '</s>'.
- Supports native bfloat16, gradient accumulation, and per-epoch checkpointing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from src.madmed_llava import (
    MedicalVQADataset,
    abstention_weighted_loss,
    load_llava_med,
    make_training_collator,
)


def filter_missing_images(dataset: MedicalVQADataset) -> MedicalVQADataset:
    """Filter out samples whose image files do not exist on disk."""
    valid_samples = [s for s in dataset.samples if os.path.exists(s.image_path)]
    removed = len(dataset.samples) - len(valid_samples)
    if removed > 0:
        print(f"[Warning] Filtering out {removed} samples with missing images")
    dataset.samples = valid_samples
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA-Med with MAD-MED.")
    parser.add_argument("--train-csv", required=True, help="Path to training CSV")
    parser.add_argument("--image-root", required=True, help="Root directory containing images")
    parser.add_argument("--output-dir", required=True, help="Output directory for fine-tuned LoRA adapter")
    parser.add_argument("--abstention-weight", type=float, default=2.0, help="Weight lambda for abstention samples (>1.0)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--no-4bit", action="store_true", default=False, help="Disable 4-bit quantization and use native bf16/fp16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for training.")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print("=" * 65)
    print("           MAD-MED LLaVA-Med Training")
    print("=" * 65)
    print(f"Device:           {torch.cuda.get_device_name(device)} (CUDA {args.gpu})")
    print(f"Train CSV:        {args.train_csv}")
    print(f"Image Root:       {args.image_root}")
    print(f"Output Dir:       {args.output_dir}")
    print(f"Abstention Weight: {args.abstention_weight}")
    print(f"Batch Size:       {args.batch_size} (Grad Accum: {args.grad_accum_steps})")
    print(f"Learning Rate:    {args.learning_rate} (Epochs: {args.epochs})")
    print(f"4-bit Quantized:  {not args.no_4bit}")
    print("-" * 65)

    print("Loading model and processor...")
    model, processor = load_llava_med(
        load_in_4bit=not args.no_4bit,
        train=True,
        device=f"cuda:{args.gpu}",
    )
    model.print_trainable_parameters()

    print("Loading dataset...")
    raw_dataset = MedicalVQADataset(args.train_csv, args.image_root)
    dataset = filter_missing_images(raw_dataset)

    abstain_count = sum(1 for s in dataset.samples if s.abstain)
    print(f"Total QA pairs:      {len(dataset)}")
    print(f"Abstention QA pairs: {abstain_count}")
    print(f"Answer QA pairs:     {len(dataset) - abstain_count}")
    print("-" * 65)

    collator = make_training_collator(processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(loader) * args.epochs
    step_count = 0
    start_time = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        print(f"\n>>> Starting Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(loader):
            abstain_mask = batch.pop("abstain_mask")
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

            outputs = model(**batch, return_dict=True)
            if outputs.loss is not None and batch["input_ids"].shape[0] == 1:
                loss = outputs.loss
                if bool(abstain_mask[0]):
                    loss = loss * args.abstention_weight
            else:
                loss = abstention_weighted_loss(
                    outputs.logits,
                    batch["labels"],
                    abstain_mask,
                    args.abstention_weight,
                )

            if not torch.isfinite(loss):
                print(f"[Warning] Non-finite loss encountered at epoch {epoch + 1}, step {step}. Skipping.")
                continue

            # Scale for gradient accumulation
            loss = loss / args.grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum_steps
            epoch_steps += 1

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1

            if (step + 1) % 50 == 0 or (step + 1) == len(loader):
                elapsed = time.time() - start_time
                steps_done = epoch * len(loader) + (step + 1)
                speed = steps_done / max(elapsed, 1e-3)
                eta_min = (total_steps - steps_done) / max(speed, 1e-3) / 60.0
                running_loss = epoch_loss / max(epoch_steps, 1)
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {step + 1}/{len(loader)} | "
                    f"Loss: {running_loss:.4f} | "
                    f"Speed: {speed:.2f} step/s | "
                    f"ETA: {eta_min:.1f}m"
                )

        mean_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f">>> Epoch {epoch + 1} Complete | Mean Loss: {mean_epoch_loss:.4f}")

        # Checkpoint per epoch
        epoch_out = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_out, exist_ok=True)
        model.save_pretrained(epoch_out)
        processor.save_pretrained(epoch_out)
        print(f"Checkpoint saved to: {epoch_out}")

    print("\nSaving final fine-tuned adapter...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Successfully saved final adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
