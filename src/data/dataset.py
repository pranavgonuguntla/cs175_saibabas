"""
data/dataset.py — PyTorch Dataset classes and DataLoader helpers for Yelp reviews.

Provides three dataset classes:

  YelpPromptedDataset
    Wraps a HuggingFace yelp_polarity split. On each __getitem__ it runs
    preprocessing, retrieves cached keywords, builds the conditioning prompt,
    and applies CFG dropout (replaces the prompt with <NULL> with probability
    cfg_drop_prob). Returns plain Python dicts with keys: prompt,
    prompt_dropped, review, label, keywords.

  MDLMYelpDataset
    Wraps a YelpPromptedDataset and tokenizes both the review (input_ids,
    attention_mask) and the prompt (keyword_ids) using the MDLM tokenizer
    (GPT-2 BPE + [MASK] + <PAD>). This is the dataset fed to the MDLM
    training and evaluation loops.

  GPT2YelpDataset / GPT2EvalDataset  (in baseline/gpt2.py)
    Equivalent wrappers for the GPT-2 fine-tuning and perplexity evaluation
    loops.

build_splits_for_sedd() is the single entry point used by all scripts to
construct train and test splits with consistent settings.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from config import (
    CFG_DROP_PROB, MAX_KEYWORDS, INCLUDE_SENTIMENT, NULL_PROMPT,
    KEYWORD_TOP_K, MAX_TRAIN_EXAMPLES, MAX_TEST_EXAMPLES, SEED,
)
from data.preprocessing import preprocess_review
from data.keywords import get_cached_keywords
from data.prompt import build_prompt


@dataclass
class PromptedExample:
    prompt: str
    prompt_dropped: str
    review: str
    label: int
    keywords: List[str]


class YelpPromptedDataset(Dataset):
    """
    Wraps a HuggingFace dataset split and produces prompted generation examples.
    Returned dict fields:
      - "prompt": conditioning string (natural language)
      - "prompt_dropped": prompt after CFG dropout (<NULL> sometimes)
      - "review": processed review text (target text)
      - "label": original label (e.g., 0/1)
      - "keywords": extracted keyword list (for debugging / evaluation)
    """
    def __init__(
        self,
        hf_split,
        max_examples: Optional[int] = None,
        cfg_drop_prob: float = 0.0,
        include_sentiment: bool = True,
        max_keywords: int = 6,
        keyword_top_k: int = 8,
        use_rake: bool = False,
        seed: int = 175,
    ):
        self.split = hf_split
        self.cfg_drop_prob = cfg_drop_prob
        self.include_sentiment = include_sentiment
        self.max_keywords = max_keywords
        self.keyword_top_k = keyword_top_k
        self.use_rake = use_rake

        self.rng = random.Random(seed)

        self.n = len(self.split) if max_examples is None else min(max_examples, len(self.split))

    def __len__(self) -> int:
        return self.n

    def _extract_keywords(self, text: str) -> List[str]:
        return get_cached_keywords(
            text,
            top_k=self.keyword_top_k,
            use_rake=self.use_rake,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.split[idx]
        label = int(ex["label"])
        review = preprocess_review(ex["text"])

        keywords = self._extract_keywords(review)
        prompt = build_prompt(
            keywords=keywords,
            label=label,
            max_keywords=self.max_keywords,
            include_sentiment=self.include_sentiment,
            natural_style=True,
        )

        # CFG condition dropout: sometimes replace prompt with NULL to train unconditional branch
        if self.cfg_drop_prob > 0 and self.rng.random() < self.cfg_drop_prob:
            prompt_dropped = NULL_PROMPT
        else:
            prompt_dropped = prompt

        return {
            "prompt": prompt,
            "prompt_dropped": prompt_dropped,
            "review": review,
            "label": label,
            "keywords": keywords,
        }


def build_splits_for_sedd(
    raw_datasets,
    max_train_examples: Optional[int] = None,
    max_test_examples: Optional[int] = None,
    cfg_drop_prob: float = 0.0,
    include_sentiment: bool = True,
    max_keywords: int = 6,
    keyword_top_k: int = 8,
    use_rake: bool = False,
    seed: int = 175,
):
    train_ds = YelpPromptedDataset(
        raw_datasets["train"],
        max_examples=max_train_examples,
        cfg_drop_prob=cfg_drop_prob,
        include_sentiment=include_sentiment,
        max_keywords=max_keywords,
        keyword_top_k=keyword_top_k,
        use_rake=use_rake,
        seed=seed,
    )
    test_ds = YelpPromptedDataset(
        raw_datasets["test"],
        max_examples=max_test_examples,
        cfg_drop_prob=0.0,  # no dropout for evaluation split object
        include_sentiment=include_sentiment,
        max_keywords=max_keywords,
        keyword_top_k=keyword_top_k,
        use_rake=use_rake,
        seed=seed,
    )
    return train_ds, test_ds


class MDLMYelpDataset(Dataset):
    def __init__(self, prompted_dataset, tokenizer, max_length=256, max_keyword_length=32):
        self.prompted_ds = prompted_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_keyword_length = max_keyword_length
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.prompted_ds)

    def __getitem__(self, idx):
        item = self.prompted_ds[idx]
        review = item["review"]
        prompt = item["prompt_dropped"]

        review_encoding = self.tokenizer(
            review, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )

        if prompt == "<NULL>":
            keyword_ids = torch.full((self.max_keyword_length,), self.pad_token_id, dtype=torch.long)
        else:
            keyword_encoding = self.tokenizer(
                prompt, truncation=True, max_length=self.max_keyword_length,
                padding="max_length", return_tensors="pt"
            )
            keyword_ids = keyword_encoding["input_ids"].squeeze()

        return {
            "input_ids": review_encoding["input_ids"].squeeze(),
            "attention_mask": review_encoding["attention_mask"].squeeze(),
            "keyword_ids": keyword_ids,
            "label": item["label"],
        }


def prompted_collate_fn(batch):
    # Keep everything as Python strings/lists for now.
    # (Later, when we align with SEDD tokenizer, this collator will tokenize and pad.)
    return {
        "prompt": [b["prompt"] for b in batch],
        "prompt_dropped": [b["prompt_dropped"] for b in batch],
        "review": [b["review"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "keywords": [b["keywords"] for b in batch],
    }
