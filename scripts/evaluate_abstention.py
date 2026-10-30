"""Evaluate abstention, hallucination, and answer generation metrics.

Computes paper metrics (Tables 1, 2, and 3):
- Abstain Evaluations: Precision (P), Recall (R), F1-score (F1), Abstention Accuracy (AA), Hallucination Rate (HR)
- Answer Evaluation: Jaccard Similarity (JS), BERTScore (BS)
- Per-question breakdown + macro-average across all questions.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.madmed_llava import is_abstention_answer


def normalize_words(text: str) -> List[str]:
    """Tokenize and normalize text into words for similarity metrics."""
    if not text:
        return []
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w]


def compute_jaccard(pred: str, gold: str) -> float:
    """Compute token-level Jaccard similarity between predicted and gold answers."""
    p_set = set(normalize_words(pred))
    g_set = set(normalize_words(gold))
    if not p_set and not g_set:
        return 1.0
    if not p_set or not g_set:
        return 0.0
    return len(p_set & g_set) / len(p_set | g_set)


def evaluate_records(
    records: List[Dict[str, str]],
    compute_bertscore: bool = True,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics across evaluated QA pairs."""
    tp = fp = fn = tn = 0
    jaccard_scores: List[float] = []
    gold_answers: List[str] = []
    pred_answers: List[str] = []

    for r in records:
        gold = r["gold"]
        pred = r["pred"]
        gold_abs = is_abstention_answer(gold)
        pred_abs = is_abstention_answer(pred)

        if gold_abs and pred_abs:
            tp += 1
        elif (not gold_abs) and pred_abs:
            fp += 1
        elif gold_abs and (not pred_abs):
            fn += 1
        else:
            tn += 1
            jaccard_scores.append(compute_jaccard(pred, gold))
            gold_answers.append(gold)
            pred_answers.append(pred)

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Section 5.1: AA measures proportion of correct decisions (both appropriate answers and abstentions)
    aa = (tp + tn) / total if total > 0 else 0.0

    # Section 5.3.4: Hallucination rate on queries requiring abstention (answering when should abstain)
    hr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    mean_js = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

    bertscore_f1 = float("nan")
    if compute_bertscore and gold_answers:
        try:
            from bert_score import score as calc_bertscore
            kwargs = {"lang": "en", "verbose": False}
            if device:
                kwargs["device"] = device
            _, _, b_f1 = calc_bertscore(pred_answers, gold_answers, **kwargs)
            bertscore_f1 = float(b_f1.mean().item())
        except Exception as e:
            print(f"[Warning] BERTScore calculation skipped: {e}")

    return {
        "P": precision,
        "R": recall,
        "F1": f1,
        "AA": aa,
        "HR": hr,
        "JS": mean_js,
        "BS": bertscore_f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAD-MED abstention and answer metrics.")
    parser.add_argument("--csv", required=True, help="Path to output CSV from evaluate_llava_med")
    parser.add_argument("--no-bertscore", action="store_true", help="Disable BERTScore computation")
    parser.add_argument("--device", default=None, help="Device for BERTScore (e.g. cuda:0)")
    parser.add_argument("--output-summary", default=None, help="Optional path to save summary CSV")
    args = parser.parse_args()

    records_by_question: Dict[str, List[Dict[str, str]]] = {}
    all_records: List[Dict[str, str]] = []

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, gold in row.items():
                if not key.endswith(" - Gold"):
                    continue
                question = key[:-7]
                pred = row.get(f"{question} - Generated", "") or ""
                gold = gold or ""

                rec = {"question": question, "gold": gold, "pred": pred}
                all_records.append(rec)
                if question not in records_by_question:
                    records_by_question[question] = []
                records_by_question[question].append(rec)

    if not all_records:
        raise ValueError(f"No valid evaluation records found in {args.csv}")

    print("=" * 70)
    print("           MAD-MED Quantitative Evaluation Results")
    print("=" * 70)
    print(f"Input Predictions CSV:  {args.csv}")
    print(f"Total Evaluated Pairs:  {len(all_records)}")

    all_metrics = evaluate_records(all_records, compute_bertscore=not args.no_bertscore, device=args.device)

    print(f"Ground-Truth Breakdown: Abstain={all_metrics['TP'] + all_metrics['FN']}, Answer={all_metrics['TN'] + all_metrics['FP']}")
    print(f"Confusion Matrix:       TP={all_metrics['TP']} (Correct Abstain), FP={all_metrics['FP']} (Over-refusal)")
    print(f"                        FN={all_metrics['FN']} (Hallucination),   TN={all_metrics['TN']} (Correct Answer)")
    print("-" * 70)
    print("Abstain Evaluations:")
    print(f"  Precision (P):            {all_metrics['P']:.4f}")
    print(f"  Recall (R):               {all_metrics['R']:.4f}")
    print(f"  F1-Score (F1):            {all_metrics['F1']:.4f}")
    print(f"  Abstention Accuracy (AA): {all_metrics['AA']:.4f}  [(TP + TN) / Total]")
    print(f"  Hallucination Rate (HR):  {all_metrics['HR']:.4f}  [FN / (TP + FN)]")
    print("Answer Evaluation (Non-Abstain Queries):")
    print(f"  Jaccard Similarity (JS):  {all_metrics['JS']:.4f}")
    print(f"  BERTScore (BS):           {all_metrics['BS']:.4f}")
    print("=" * 70)

    # Per-Question Metrics Breakdown
    print("\nPer-Question Breakdown:")
    header = f"{'Question':<55} | {'P':<6} | {'R':<6} | {'F1':<6} | {'AA':<6} | {'HR':<6} | {'JS':<6} | {'BS':<6}"
    print(header)
    print("-" * len(header))

    question_summaries = []
    for q, recs in records_by_question.items():
        q_metrics = evaluate_records(recs, compute_bertscore=not args.no_bertscore, device=args.device)
        q_short = (q[:52] + "...") if len(q) > 55 else q
        print(
            f"{q_short:<55} | "
            f"{q_metrics['P']:.4f} | "
            f"{q_metrics['R']:.4f} | "
            f"{q_metrics['F1']:.4f} | "
            f"{q_metrics['AA']:.4f} | "
            f"{q_metrics['HR']:.4f} | "
            f"{q_metrics['JS']:.4f} | "
            f"{q_metrics['BS']:.4f}"
        )
        question_summaries.append({"question": q, **q_metrics})

    if args.output_summary:
        summary_dir = os.path.dirname(args.output_summary)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(args.output_summary, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Score"])
            writer.writerow(["Precision", f"{all_metrics['P']:.4f}"])
            writer.writerow(["Recall", f"{all_metrics['R']:.4f}"])
            writer.writerow(["F1", f"{all_metrics['F1']:.4f}"])
            writer.writerow(["Abstention Accuracy", f"{all_metrics['AA']:.4f}"])
            writer.writerow(["Hallucination Rate", f"{all_metrics['HR']:.4f}"])
            writer.writerow(["Jaccard Similarity", f"{all_metrics['JS']:.4f}"])
            writer.writerow(["BERTScore", f"{all_metrics['BS']:.4f}"])
        print(f"\nSummary saved to: {args.output_summary}")


if __name__ == "__main__":
    main()
