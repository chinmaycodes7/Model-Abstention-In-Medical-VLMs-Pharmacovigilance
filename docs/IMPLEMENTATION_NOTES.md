# MAD-MED v3 Implementation & Final Results

## 1. Problem Diagnosis in v2 Run

### Symptoms
In the full 3-epoch v2 training run and subsequent 504-sample test evaluation:
- **Precision:** 1.0000
- **Recall:** 0.0013 (2 True Positives out of 1,549 abstention targets)
- **Hallucination Rate:** 0.9987 (1,547 False Negatives)
- The model generated hallucinated medical dosages, diagnoses, and adverse events instead of outputting the learned abstention phrase `ABSTAIN: Cannot be Determined from the provided Context`.

### Root Cause: Token-Boundary Off-by-One Masking in Collator
1. In `src/madmed_llava_v2.py`, `build_prompt(...)` produced a string ending with `"Answer: "` (including a trailing space).
2. Under the Mistral/Llama SentencePiece tokenizer, `"Answer: "` in isolation tokenizes the space as an explicit standalone token:
   `['\n', 'An', 'swer', ':', ' ']` (length = 4).
3. In contrast, when concatenated as `prompt + answer` (`"Answer: ABSTAIN..."`), SentencePiece merges the space into the subsequent token:
   `[':', ' AB', 'ST', 'AIN']`.
4. As a result, when the collator computed `prompt_len` from the isolated prompt and applied `labels[:prompt_len] = -100`, it masked out the first token of the target answer (`' AB'`).
5. For abstention samples, the model was trained exclusively on `'STAIN: Cannot be Determined...'`. It received zero supervision on predicting `' AB'` right after `"Answer: "`.
6. At test time, when prompted with `"Answer: "`, the model lacked the trained transition to emit `' AB'`, causing it to fall back to generating pretrained clinical terminology.

---

## 2. Changes Implemented in v3

### `src/madmed_llava_v3.py`
- **Clean Prompt Terminator:** `build_prompt(...)` ends with `\nAnswer:` with no trailing whitespace.
- **Explicit Delimiter Search in Collator:**
  Instead of relying on heuristic sequence lengths, `make_training_collator` directly searches for the tokenized `\nAnswer:` delimiter (`[13, 2820, 16981, 28747]`) in `input_ids`.
- **Complete Answer Supervision:**
  - All tokens up to and including `\nAnswer:` are masked with `-100`.
  - Every single token of the response starting with `' AB'` and concluding with `'</s>'` (EOS) receives active gradient loss.
- **Preserved LoRA Backbone:** Same parameter-efficient configuration (Rank 16, Alpha 32, all projection layers).

### `scripts/train_madmed_llava_med_v3.py`
- Full 3-epoch fine-tuning script with sample-level weighted loss $\lambda \cdot \mathcal{L}$ ($\lambda = 2.0$) on abstention samples.
- Saves per-epoch checkpoints to `adapters/llava-med-madmed-v3/epoch_{1,2,3}`.
- Outputs final fine-tuned adapter to `adapters/llava-med-madmed-v3`.

### `scripts/evaluate_llava_med_v3.py`
- Controlled evaluation script for `image_text`, `image`, and `text` modes.
- Generates answers matching the v3 prompt format.
- Outputs predictions to `outputs/full_eval_image_text_v3.csv`.

### `scripts/evaluate_abstention_v3.py`
- Evaluates full test predictions against gold annotations for Precision, Recall, F1, Abstention Accuracy (AA), Hallucination Rate (HR), Jaccard Similarity (JS), and BERTScore (BS).
- Outputs summary to `outputs/summary_image_text_v3.csv`.

---

## 3. Final Empirical Results Comparison (Multimodal `Image+Text`)

Evaluated on all 504 test samples (3,024 question queries) from `task1_VQA_test_504.csv`:

| Metric | Paper (Table 3 Post-MAD-MED) | Previous v2 Run | MAD-MED v3 Empirical Run | Status / Outcome |
| :--- | :---: | :---: | :---: | :--- |
| **Precision ($P$)** | **0.782** | 1.0000 | **0.9692** | High reliability; very low false abstention rate ($FP = 48$). |
| **Recall ($R$)** | **0.785** | 0.0013 | **0.9755** | Successfully resolved; model abstains on 1,511 / 1,549 queries. |
| **$F_1$-Score ($F_1$)** | **0.783** | 0.0026 | **0.9723** | Dramatic improvement from 0.0026 $\rightarrow$ 0.9723. |
| **Abstention Accuracy ($\text{AA}$)** | **0.813** | 0.4884 | **0.9716** | 2,938 / 3,024 decisions are correct. |
| **Hallucination Rate ($\text{HR}$)** | **0.237** | 0.9987 | **0.0245** | Hallucinations suppressed from 99.8% down to 2.45%. |
| **BERTScore ($\text{BS}$)** | **0.894** | 0.5356 | **0.8848** | Closely matches paper (0.8848 vs. 0.894). |
| **Jaccard Similarity ($\text{JS}$)** | **0.689** | 0.0746 | **0.3770** | Increased from 0.0746 $\rightarrow$ 0.3770. |

---

## 4. Per-Question Breakdown (v3)

| Question | P | R | F1 | AA | HR | JS | BS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **What drug is the patient using?** | 0.9596 | 0.9223 | 0.9406 | 0.9762 | 0.0777 | 0.1575 | 0.8322 |
| **What is the route and pattern of drug administration?** | 0.9752 | 0.9937 | 0.9843 | 0.9702 | 0.0063 | 0.4444 | 0.9546 |
| **What adverse drug events are described in the case?** | 0.7609 | 0.7292 | 0.7447 | 0.9524 | 0.2708 | 0.2720 | 0.8688 |
| **What psychological medical conditions are described in the image?** | 0.9980 | 0.9960 | 0.9970 | 0.9940 | 0.0040 | 0.0000 | N/A |
| **Which body part is affected in the image?** | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.6060 | 0.9374 |
| **Are the adverse events acute or chronic based on the information provided?** | 0.9534 | 0.9715 | 0.9624 | 0.9365 | 0.0285 | 0.6508 | 0.8886 |
