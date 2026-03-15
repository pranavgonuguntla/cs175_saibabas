"""
GPT-2 zero-shot sanity checker.
Generates reviews from prompts using pretrained GPT-2 to establish a baseline.
Run: python baseline/sanity_check.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import (
    DEVICE, SEED, CFG_DROP_PROB, INCLUDE_SENTIMENT, MAX_KEYWORDS, KEYWORD_TOP_K,
    MAX_TRAIN_EXAMPLES, MAX_TEST_EXAMPLES, GPT2_MODEL_NAME,
    MAX_NEW_TOKENS, NUM_SAMPLES, GENERATION_SEED, LABEL_TO_SENTIMENT, YELP_DATASET_NAME,
)
from datasets import load_dataset
from data.dataset import build_splits_for_sedd
from data.prompt import build_prompt
from baseline.gpt2 import load_gpt2, generate_review_gpt2


def main():
    print("=" * 80)
    print("GPT-2 BASELINE SANITY CHECKER")
    print("=" * 80)

    raw_datasets = load_dataset(YELP_DATASET_NAME)

    train_prompted_ds, test_prompted_ds = build_splits_for_sedd(
        raw_datasets=raw_datasets,
        max_train_examples=MAX_TRAIN_EXAMPLES,
        max_test_examples=MAX_TEST_EXAMPLES,
        cfg_drop_prob=CFG_DROP_PROB,
        include_sentiment=INCLUDE_SENTIMENT,
        max_keywords=MAX_KEYWORDS,
        keyword_top_k=KEYWORD_TOP_K,
        use_rake=False,
        seed=SEED,
    )

    gpt2_tokenizer, gpt2_model = load_gpt2(GPT2_MODEL_NAME, DEVICE)

    # Set seed for reproducibility
    torch.manual_seed(GENERATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GENERATION_SEED)

    # --- Zero-shot generation ---
    print("\n" + "=" * 80)
    print("ZERO-SHOT GENERATION SAMPLES (Pretrained GPT-2)")
    print("=" * 80)

    for i in range(NUM_SAMPLES):
        print(f"\n{'=' * 80}")
        print(f"SAMPLE {i+1}/{NUM_SAMPLES}")
        print(f"{'=' * 80}")

        sample = test_prompted_ds[i]
        prompt = sample["prompt"]
        true_review = sample["review"]
        label = "positive" if sample["label"] == 1 else "negative"

        print(f"\nPrompt: {prompt}")
        print(f"\nTrue Label: {label}")
        print(f"\nTrue Review (first 300 chars):\n{true_review[:300]}...")

        print(f"\n{'-' * 80}")
        print("GPT-2 GENERATION (Temperature=0.8, Top-p=0.95):")
        print(f"{'-' * 80}")

        generated = generate_review_gpt2(
            prompt=prompt,
            tokenizer=gpt2_tokenizer,
            model=gpt2_model,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            device=DEVICE,
        )
        print(generated[0])

        print(f"\n{'-' * 80}")
        print("GPT-2 GENERATION (Greedy Decoding):")
        print(f"{'-' * 80}")

        generated_greedy = generate_review_gpt2(
            prompt=prompt,
            tokenizer=gpt2_tokenizer,
            model=gpt2_model,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            device=DEVICE,
        )
        print(generated_greedy[0])

    # --- Sentiment conditioning test ---
    print("\n" + "=" * 80)
    print("SENTIMENT CONDITIONING TEST")
    print("=" * 80)

    test_keywords = ["food", "service", "atmosphere"]

    for sentiment_label in [0, 1]:
        sentiment = LABEL_TO_SENTIMENT[sentiment_label]
        test_prompt = build_prompt(
            keywords=test_keywords,
            label=sentiment_label,
            max_keywords=6,
            include_sentiment=True,
            natural_style=True
        )

        print(f"\n{'-' * 80}")
        print(f"Prompt ({sentiment}): {test_prompt}")
        print(f"{'-' * 80}")

        gen = generate_review_gpt2(
            prompt=test_prompt,
            tokenizer=gpt2_tokenizer,
            model=gpt2_model,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            device=DEVICE,
        )
        print(gen[0])

    print("\n" + "=" * 80)
    print("GPT-2 SANITY CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
