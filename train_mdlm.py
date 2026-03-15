"""
Train the MDLM (Masked Diffusion Language Model) on the Yelp dataset.
Run: python train_mdlm.py
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from config import DEVICE, SEED, CFG_DROP_PROB, INCLUDE_SENTIMENT, MAX_KEYWORDS, KEYWORD_TOP_K, MAX_TRAIN_EXAMPLES, MAX_TEST_EXAMPLES
from datasets import load_dataset
from data.dataset import build_splits_for_sedd, MDLMYelpDataset
from data.keywords import flush_keyword_cache
from model.transformer import MDLMTransformer
from model.diffusion import train_step, sample_mdlm


def main():
    print("=" * 80)
    print("MDLM TRAINING")
    print("=" * 80)

    # Load dataset
    from config import YELP_DATASET_NAME
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

    print("1. Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'mask_token': '[MASK]', 'pad_token': '<PAD>'})
    vocab_size = len(tokenizer)
    print(f"    Vocab size: {vocab_size}")
    print(f"    [MASK] ID: {tokenizer.mask_token_id}")

    print("\n2. Creating datasets for test...")

    full_train_dataset = MDLMYelpDataset(
        train_prompted_ds, tokenizer, max_length=128, max_keyword_length=32
    )
    full_val_dataset = MDLMYelpDataset(
        test_prompted_ds, tokenizer, max_length=128, max_keyword_length=32
    )

    train_subset_size = MAX_TRAIN_EXAMPLES
    val_subset_size   = MAX_TEST_EXAMPLES

    train_indices = list(range(min(train_subset_size, len(full_train_dataset))))
    val_indices   = list(range(min(val_subset_size,   len(full_val_dataset))))

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset   = Subset(full_val_dataset,   val_indices)

    print(f"    Train: {len(train_dataset)} samples")
    print(f"    Val:   {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")

    print("\n3. Creating model...")
    model = MDLMTransformer(
        vocab_size=vocab_size,
        d_model=768,
        nhead=12,
        num_layers=8,
        dim_feedforward=3072,
        dropout=0.1,
        max_seq_len=128,
    )
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    num_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n4. Training for {num_epochs} epochs...")
    print("=" * 80)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)

        model.train()
        train_losses, train_accs = [], []

        pbar = tqdm(train_loader, desc="Training", leave=True)
        for batch in pbar:
            loss, acc = train_step(
                model, batch, optimizer, tokenizer.mask_token_id, 'cosine', DEVICE
            )
            train_losses.append(loss)
            train_accs.append(acc)
            pbar.set_postfix({'loss': f"{loss:.4f}", 'acc': f"{acc:.3f}"})

        avg_train_loss = np.mean(train_losses)
        avg_train_acc  = np.mean(train_accs)

        model.eval()
        val_losses, val_accs = [], []

        import torch.nn.functional as F
        from model.diffusion import get_mask_prob, create_masked_input

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluation"):
                clean_tokens   = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                keyword_ids    = batch["keyword_ids"].to(DEVICE)
                batch_size, seq_len = clean_tokens.shape

                t = torch.rand(batch_size, 1, device=DEVICE)
                mask_probs = torch.tensor(
                    [get_mask_prob(t_i.item(), 'cosine') for t_i in t], device=DEVICE
                ).view(batch_size, 1)

                noisy_input, mask_indices = create_masked_input(
                    clean_tokens, tokenizer.mask_token_id, mask_probs, attention_mask
                )

                logits = model(noisy_input, condition_tokens=keyword_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), clean_tokens.view(-1), reduction='none'
                )
                loss = loss.view(batch_size, seq_len)
                masked_loss = (loss * mask_indices.float()).sum() / (mask_indices.sum() + 1e-8)

                predictions = logits.argmax(dim=-1)
                correct     = (predictions == clean_tokens) & mask_indices
                accuracy    = correct.sum().float() / (mask_indices.sum() + 1e-8)

                val_losses.append(masked_loss.item())
                val_accs.append(accuracy.item())

        avg_val_loss = np.mean(val_losses)
        avg_val_acc  = np.mean(val_accs)

        scheduler.step()

        print(f"\n  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.3f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.3f}")

    print("\n" + "=" * 80)
    print("5. Testing Generation")
    print("=" * 80)

    model.eval()
    torch.save(model.state_dict(), "mdlm_model.pth")
    print("    Model saved to mdlm_model.pth")
    flush_keyword_cache()

    # 5 diverse keyword prompts covering different sentiments and aspects
    test_keywords = [
        "positive great food delicious",
        "negative bad service rude",
        "positive amazing atmosphere cozy",
        "negative disappointing cold food",
        "positive friendly staff excellent",
    ]

    # 10 guidance scales: from no guidance → very strong guidance
    guidance_scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

    results = {}

    for keywords in test_keywords:
        print(f"\n{'='*80}")
        print(f"KEYWORDS: {keywords}")
        print(f"{'='*80}")
        results[keywords] = {}

        for scale in guidance_scales:
            text = sample_mdlm(
                model, tokenizer,
                keywords=keywords,
                n_steps=150,
                guidance_scale=scale,
                max_length=128,
                device=DEVICE
            )
            results[keywords][scale] = text
            print(f"\n  guidance={scale:<5} | {text[:120]}{'...' if len(text) > 120 else ''}")

    print("\n" + "=" * 80)
    print("6. RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Keyword':<40} {'Scale':<8} {'Preview (first 80 chars)'}")
    print("-" * 80)

    for keywords, scale_results in results.items():
        keyword_short = keywords[:38]
        for scale, text in scale_results.items():
            preview = text[:78].replace('\n', ' ')
            print(f"{keyword_short:<40} {scale:<8} {preview}")
        print()

    print("=" * 80)
    print("7. GUIDANCE SCALE EFFECT (first keyword only)")
    print("=" * 80)
    first_keyword = test_keywords[0]
    print(f"\nKeyword: '{first_keyword}'\n")
    print(f"{'Scale':<8} | {'Observation'}")
    print("-" * 50)

    scale_labels = {
        0.5:  "BELOW 1 — pushes AWAY from keywords",
        1.0:  "=1 — free generation, keywords ignored",
        1.5:  "slight keyword influence",
        2.0:  "moderate keyword influence",
        2.5:  "noticeable keyword steering",
        3.0:  "strong keyword influence (recommended)",
        4.0:  "very strong steering",
        5.0:  "heavy steering, may reduce fluency",
        7.0:  "aggressive steering",
        10.0: "maximum steering, likely repetitive",
    }
    for scale in guidance_scales:
        print(f"  {scale:<6} | {scale_labels[scale]}")
        print(f"         | {results[first_keyword][scale][:90]}")
        print()

    return model, tokenizer


if __name__ == "__main__":
    main()
