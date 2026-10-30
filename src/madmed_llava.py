"""Core utilities for the LLaVA-Med MAD-MED implementation.

Features:
- Exact Prompt-Answer Token Boundary Alignment:
  - Prompt ends cleanly with '\\nAnswer:' (no trailing whitespace).
  - Answer begins with a leading space and ends with EOS token.
  - Collator accurately locates the exact '\\nAnswer:' token sequence in input_ids,
    ensuring that 100% of answer tokens (including the crucial leading ' AB' token
    for abstention, and the terminal '</s>' token) receive loss supervision.
- Section 4.4 canonical system prompt incorporated into prompts.
- Parameter-efficient LoRA fine-tuning on frozen vision-language backbone.
- Sample-level abstention-weighted loss: lambda * CE for abstention, CE otherwise.
- Autoregressive generation without heuristic thresholds or confidence cutoffs.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
ABSTAIN_TEXT = "ABSTAIN: Cannot be Determined from the provided Context"
SYSTEM_PROMPT = (
    "You are a medical visual question answering assistant. "
    "Your task is to answer the user’s question using only the information that is explicitly present "
    "in the provided image and/or text. If the question cannot be answered, you must respond with: "
    "‘ABSTAIN: Cannot be Determined from the provided Context’. Do not guess. Do not assume missing "
    "details. Do not provide speculative or uncertain medical advice."
)

DEFAULT_QUESTIONS = [
    "What drug is the patient using?",
    "What is the route and pattern of drug administration?",
    "What adverse drug events are described in the case?",
    "What psychological medical conditions are described in the image?",
    "Which body part is affected in the image?",
    "Are the adverse events acute or chronic based on the information provided?",
]


def is_abstention_answer(answer: str) -> bool:
    """Recognize the abstention labels used in the recovered datasets."""
    if not answer:
        return False
    normalized = " ".join(answer.strip().lower().split())
    return (
        normalized.startswith("abstain")
        or "cannot be determined" in normalized
        or "cannot be determined from the provided" in normalized
    )


def normalize_answer(answer: str) -> str:
    """Map legacy abstention wording to the canonical paper response."""
    answer = answer.strip()
    return ABSTAIN_TEXT if is_abstention_answer(answer) else answer


def get_processor(model_id: str = MODEL_ID) -> LlavaProcessor:
    """Safely load processor with fast-tokenizer failure handling."""
    try:
        return AutoProcessor.from_pretrained(model_id, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        image_processor = CLIPImageProcessor.from_pretrained(model_id)
        return LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)


def load_llava_med(
    load_in_4bit: bool = True,
    train: bool = False,
    device: Optional[str] = None,
):
    """Load the LLaVA-Med model and processor."""
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(target_device)

    processor = get_processor(MODEL_ID)

    if train:
        model.config.use_cache = False
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        else:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    return model, processor


def load_finetuned_llava_med(adapter_path: str, load_in_4bit: bool = True, device: Optional[str] = None):
    """Load the base LLaVA-Med model with a saved LoRA adapter."""
    base_model, processor = load_llava_med(load_in_4bit=load_in_4bit, train=False, device=device)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, processor


def build_prompt(
    processor,
    question: str,
    text_context: Optional[str] = None,
    image: bool = True,
    include_system_prompt: bool = True,
) -> str:
    """Construct prompt for LLaVA-Med combining optional image, system prompt, context, and question."""
    parts = []
    if image:
        parts.append("<image>")
    if include_system_prompt:
        parts.append(SYSTEM_PROMPT)
    if text_context and text_context.strip():
        parts.append(f"Context:\n{text_context.strip()}")
    parts.append(f"Question:\n{question.strip()}")
    parts.append("Answer:")
    return "\n".join(parts)


@dataclass
class Sample:
    image_path: str
    text: str
    question: str
    answer: str
    abstain: bool


class MedicalVQADataset(Dataset):
    """Flatten an Image/Text + question-column CSV into QA training examples."""

    def __init__(self, csv_path: str, image_base_path: str):
        self.samples: List[Sample] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = [fn.strip().lstrip("\ufeff") for fn in (reader.fieldnames or []) if fn and fn.strip()]
            if "Image" not in fields or "Text" not in fields:
                raise ValueError("CSV must contain 'Image' and 'Text' columns")
            question_cols = [c for c in fields if c not in ("Image", "Text")]

            for raw_row in reader:
                row = {
                    k.strip().lstrip("\ufeff"): (v or "")
                    for k, v in raw_row.items()
                    if k is not None and k.strip()
                }
                image_path = os.path.join(image_base_path, row["Image"].lstrip("/\\"))
                text = row.get("Text", "")
                for question in question_cols:
                    raw_answer = row.get(question, "").strip()
                    if not raw_answer:
                        continue
                    answer = normalize_answer(raw_answer)
                    self.samples.append(
                        Sample(
                            image_path=image_path,
                            text=text,
                            question=question,
                            answer=answer,
                            abstain=is_abstention_answer(raw_answer),
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        return {
            "image": image,
            "text": s.text,
            "question": s.question,
            "answer": s.answer,
            "abstain": s.abstain,
        }


def make_training_collator(processor):
    """Create a collator with robust prompt masking and sample-level abstention labels."""
    tok = processor.tokenizer
    eos = tok.eos_token or "</s>"
    ans_marker = tok.encode("\nAnswer:", add_special_tokens=False)
    ans_marker_len = len(ans_marker)

    def collate(batch: List[Dict]):
        images = [b["image"] for b in batch]
        full_texts = []
        for b in batch:
            prompt = build_prompt(
                processor,
                question=b["question"],
                text_context=b["text"],
                image=True,
            )
            # Leading space so first answer token attaches naturally; trailing EOS for termination
            full_texts.append(f"{prompt} {b['answer'].strip()}{eos}")

        enc = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = enc["input_ids"].clone()
        for i in range(len(batch)):
            seq = labels[i].tolist()
            # Find the last occurrence of '\nAnswer:' marker
            match_idx = -1
            for j in range(len(seq) - ans_marker_len, -1, -1):
                if seq[j : j + ans_marker_len] == ans_marker:
                    match_idx = j + ans_marker_len
                    break

            if match_idx != -1:
                labels[i, :match_idx] = -100
            else:
                # Fallback: mask up to halfway if marker not found
                labels[i, : len(seq) // 2] = -100

        # Padding must never contribute to the loss
        if "attention_mask" in enc:
            labels[enc["attention_mask"] == 0] = -100

        enc["labels"] = labels
        enc["abstain_mask"] = torch.tensor(
            [bool(b["abstain"]) for b in batch], dtype=torch.bool
        )
        return enc

    return collate


def move_batch_to_device(batch, device):
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
        if k != "abstain_mask"
    }


def abstention_weighted_loss(logits: torch.Tensor, labels: torch.Tensor, abstain_mask: torch.Tensor, weight: float):
    """MAD-MED objective: lambda * CE for abstention samples, CE otherwise."""
    if weight <= 1.0:
        raise ValueError("abstention weight must be > 1, as specified by the paper")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)

    valid = shift_labels.ne(-100)
    denom = valid.sum(dim=1).clamp_min(1)
    per_sample_loss = (token_loss * valid).sum(dim=1) / denom

    weights = torch.ones_like(per_sample_loss)
    weights = weights + (weight - 1.0) * abstain_mask.to(weights.device, dtype=weights.dtype)
    return (per_sample_loss * weights).mean()


def generate_answer(
    model,
    processor,
    question: str,
    image_path: Optional[str] = None,
    text_context: Optional[str] = None,
    max_new_tokens: int = 40,
):
    """Generate without any confidence threshold or inference-time abstention rule."""
    image = None
    image_input = image_path is not None
    if image_input:
        image = Image.open(image_path).convert("RGB")

    prompt = build_prompt(
        processor,
        question=question,
        text_context=text_context,
        image=image_input,
    )
    kwargs = {"text": prompt, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = [image]

    inputs = processor(**kwargs)
    model_dtype = next(model.parameters()).dtype
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if key == "pixel_values":
                inputs[key] = value.to(model.device, dtype=model_dtype)
            else:
                inputs[key] = value.to(model.device)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    eos_id = processor.tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated = processor.tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()
    return generated
