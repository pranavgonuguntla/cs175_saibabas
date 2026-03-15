import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from config import DEVICE, GPT2_MODEL_NAME, NULL_PROMPT


def load_gpt2(model_name: str = GPT2_MODEL_NAME, device: str = DEVICE):
    """Load GPT-2 tokenizer and model."""
    print(f"\nLoading GPT-2 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return tokenizer, model


def generate_review_gpt2(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    device: str = DEVICE,
) -> List[str]:
    """
    Generate review text from a prompt using GPT-2.

    Args:
        prompt: The conditioning prompt (e.g., "negative korean bbq dry service")
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        num_return_sequences: Number of sequences to generate
        do_sample: Whether to use sampling (False = greedy)

    Returns:
        List of generated review texts
    """
    # Format prompt for GPT-2 (add explicit separator)
    formatted_prompt = f"{prompt}\n\nReview:"

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_texts = []
    for output in outputs:
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        # Remove the prompt part
        review_text = full_text.split("Review:", 1)[-1].strip()
        generated_texts.append(review_text)

    return generated_texts


class GPT2YelpDataset(Dataset):
    """
    Dataset that formats prompted reviews for GPT-2 causal LM training.
    Format: "{prompt}\n\nReview: {review_text}<|endoftext|>"
    """
    def __init__(self, prompted_dataset, tokenizer, max_length=256):
        self.prompted_ds = prompted_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompted_ds)

    def __getitem__(self, idx):
        item = self.prompted_ds[idx]

        # Use prompt_dropped to leverage CFG training data
        prompt = item["prompt_dropped"]
        review = item["review"]

        # Create training text
        if prompt == NULL_PROMPT:
            # Unconditional generation (no prompt)
            text = f"Review: {review}"
        else:
            # Conditional generation (with prompt)
            text = f"{prompt}\n\nReview: {review}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # For causal LM, labels = input_ids
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


class GPT2EvalDataset(Dataset):
    def __init__(self, prompted_dataset, tokenizer, max_length=128):
        self.ds = prompted_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        review = self.ds[idx]["review"]

        enc = self.tokenizer(
            review,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return enc["input_ids"].squeeze()
