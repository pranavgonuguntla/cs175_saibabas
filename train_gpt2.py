"""
Fine-tune GPT-2 on the Yelp dataset as a baseline.
Run: python train_gpt2.py
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

from config import (
    DEVICE, SEED, CFG_DROP_PROB, INCLUDE_SENTIMENT, MAX_KEYWORDS, KEYWORD_TOP_K,
    MAX_TRAIN_EXAMPLES, MAX_TEST_EXAMPLES,
    GPT2_MODEL_NAME, FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, FINETUNE_LR,
    FINETUNE_MAX_LENGTH, FINETUNE_OUTPUT_DIR, YELP_DATASET_NAME,
)
from datasets import load_dataset
from data.dataset import build_splits_for_sedd
from baseline.gpt2 import load_gpt2, GPT2YelpDataset


def main():
    print("=" * 80)
    print("GPT-2 FINE-TUNING ON YELP REVIEWS")
    print("=" * 80)

    # Load dataset
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

    print("\nPreparing training data...")
    gpt2_train_dataset = GPT2YelpDataset(
        prompted_dataset=train_prompted_ds,
        tokenizer=gpt2_tokenizer,
        max_length=FINETUNE_MAX_LENGTH
    )
    gpt2_train_dataset = Subset(gpt2_train_dataset, range(50000))

    gpt2_eval_dataset = GPT2YelpDataset(
        prompted_dataset=test_prompted_ds,
        tokenizer=gpt2_tokenizer,
        max_length=FINETUNE_MAX_LENGTH
    )

    print(f"Training samples: {len(gpt2_train_dataset)}")
    print(f"Eval samples: {len(gpt2_eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=FINETUNE_OUTPUT_DIR,
        num_train_epochs=FINETUNE_EPOCHS,
        per_device_train_batch_size=FINETUNE_BATCH_SIZE,
        per_device_eval_batch_size=FINETUNE_BATCH_SIZE,
        learning_rate=FINETUNE_LR,
        warmup_steps=100,
        logging_steps=50,
        eval_steps=200,
        save_steps=600,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    print("\nInitializing trainer...")
    trainer = Trainer(
        model=gpt2_model,
        args=training_args,
        train_dataset=gpt2_train_dataset,
        eval_dataset=gpt2_eval_dataset,
        processing_class=gpt2_tokenizer,
    )

    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING...")
    print("=" * 80)
    print(f"Training for {FINETUNE_EPOCHS} epochs on {len(gpt2_train_dataset)} examples")
    print(f"Batch size: {FINETUNE_BATCH_SIZE}, Learning rate: {FINETUNE_LR}")
    print("=" * 80 + "\n")

    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    trainer.save_model(FINETUNE_OUTPUT_DIR)
    gpt2_tokenizer.save_pretrained(FINETUNE_OUTPUT_DIR)
    print(f"\nModel saved to: {FINETUNE_OUTPUT_DIR}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"\nTest Loss: {eval_results['eval_loss']:.4f}")
    print(f"Test Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

    # Test generation with fine-tuned model
    from baseline.gpt2 import generate_review_gpt2

    print("\n" + "=" * 80)
    print("TESTING FINE-TUNED MODEL GENERATION")
    print("=" * 80)

    for i in range(3):
        sample = test_prompted_ds[i]
        prompt = sample["prompt"]

        print(f"\n{'=' * 80}")
        print(f"FINE-TUNED SAMPLE {i+1}")
        print(f"{'=' * 80}")
        print(f"Prompt: {prompt}\n")

        generated = generate_review_gpt2(
            prompt=prompt,
            tokenizer=gpt2_tokenizer,
            model=gpt2_model,
            max_new_tokens=120,
            temperature=0.8,
            do_sample=True,
            device=DEVICE,
        )

        print(f"Generated Review:\n{generated[0]}")
        print(f"\nTrue Review:\n{sample['review'][:300]}...")

    print("\n" + "=" * 80)
    print("Fine-tuning complete! Compare these outputs with pretrained GPT-2.")
    print("=" * 80)


if __name__ == "__main__":
    main()
