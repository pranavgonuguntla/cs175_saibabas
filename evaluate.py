"""
Full evaluation suite: perplexity, ROUGE, BERTScore, and sentiment accuracy.
Loads saved MDLM model and fine-tuned GPT-2 and compares them.

Run: python evaluate.py [--mdlm_path mdlm_model.pth] [--gpt2_path gpt2_yelp_finetuned]
"""
import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
from tqdm import tqdm

from config import DEVICE, SEED, CFG_DROP_PROB, INCLUDE_SENTIMENT, MAX_KEYWORDS, KEYWORD_TOP_K, MAX_TRAIN_EXAMPLES, MAX_TEST_EXAMPLES, YELP_DATASET_NAME
from datasets import load_dataset
from data.dataset import build_splits_for_sedd, MDLMYelpDataset
from model.transformer import MDLMTransformer
from model.diffusion import sample_mdlm, get_mask_prob, create_masked_input
from baseline.gpt2 import GPT2EvalDataset, generate_review_gpt2
from evaluation.metrics import compute_rouge


def load_mdlm_model(model_path: str, vocab_size: int, device: str):
    model = MDLMTransformer(
        vocab_size=vocab_size,
        d_model=768,
        nhead=12,
        num_layers=8,
        dim_feedforward=3072,
        max_seq_len=128
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model


def eval_mdlm_perplexity(model, tokenizer, eval_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="MDLM perplexity"):
            input_ids = batch["input_ids"].to(device)
            keyword_ids = batch["keyword_ids"].to(device)

            logits = model(input_ids, condition_tokens=keyword_ids)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                ignore_index=pad_id,
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += (input_ids != pad_id).sum().item()

    mdlm_loss = total_loss / total_tokens
    mdlm_perplexity = math.exp(mdlm_loss)
    return mdlm_loss, mdlm_perplexity


def eval_gpt2_perplexity(gpt2_model, gpt2_tokenizer, gpt2_loader, device):
    gpt2_model.eval()
    total_loss = 0
    total_tokens = 0
    pad_id = gpt2_tokenizer.pad_token_id

    with torch.no_grad():
        for input_ids in tqdm(gpt2_loader, desc="GPT-2 perplexity"):
            input_ids = input_ids.to(device)
            outputs = gpt2_model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            tokens = (input_ids != pad_id).sum()
            total_loss += loss.item() * tokens.item()
            total_tokens += tokens.item()

    gpt2_loss = total_loss / total_tokens
    gpt2_perplexity = math.exp(gpt2_loss)
    return gpt2_loss, gpt2_perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm_path", default="mdlm_model.pth")
    parser.add_argument("--gpt2_path", default="gpt2_yelp_finetuned")
    parser.add_argument("--n_rouge", type=int, default=10)
    parser.add_argument("--n_sentiment", type=int, default=200)
    args = parser.parse_args()

    # Load dataset
    raw_datasets = load_dataset(YELP_DATASET_NAME)
    train_prompted_ds, test_prompted_ds = build_splits_for_sedd(
        raw_datasets=raw_datasets,
        max_train_examples=MAX_TRAIN_EXAMPLES,
        max_test_examples=MAX_TEST_EXAMPLES,
        cfg_drop_prob=0.0,
        include_sentiment=INCLUDE_SENTIMENT,
        max_keywords=MAX_KEYWORDS,
        keyword_top_k=KEYWORD_TOP_K,
        use_rake=False,
        seed=SEED,
    )

    # Load MDLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'mask_token': '[MASK]', 'pad_token': '<PAD>'})
    vocab_size = len(tokenizer)

    # Load MDLM model
    print(f"\nLoading MDLM from {args.mdlm_path}")
    mdlm_model = load_mdlm_model(args.mdlm_path, vocab_size, DEVICE)

    # Load fine-tuned GPT-2
    print(f"Loading fine-tuned GPT-2 from {args.gpt2_path}")
    gpt2_tokenizer = AutoTokenizer.from_pretrained(args.gpt2_path)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained(args.gpt2_path).to(DEVICE)
    gpt2_model.eval()

    # Build eval datasets
    eval_dataset = MDLMYelpDataset(test_prompted_ds, tokenizer, max_length=128)
    small_eval = Subset(eval_dataset, range(1000))
    eval_loader = DataLoader(small_eval, batch_size=16, shuffle=False)

    gpt2_eval_dataset = GPT2EvalDataset(test_prompted_ds, gpt2_tokenizer, max_length=128)
    small_gpt2_eval = Subset(gpt2_eval_dataset, range(1000))
    gpt2_loader = DataLoader(small_gpt2_eval, batch_size=16, shuffle=False)

    # --- Perplexity ---
    mdlm_loss, mdlm_perplexity = eval_mdlm_perplexity(mdlm_model, tokenizer, eval_loader, DEVICE)
    gpt2_loss, gpt2_perplexity = eval_gpt2_perplexity(gpt2_model, gpt2_tokenizer, gpt2_loader, DEVICE)

    print("\nMDLM Diffusion Evaluation Results")
    print("----------------------------------")
    print(f"Loss: {mdlm_loss:.4f}")
    print(f"Pseudo-Perplexity: {mdlm_perplexity:.2f}")

    print("\nGPT-2 Evaluation Results")
    print("------------------------")
    print(f"Loss: {gpt2_loss:.4f}")
    print(f"Perplexity: {gpt2_perplexity:.2f}")

    # --- ROUGE ---
    references = [test_prompted_ds[i]["review"] for i in range(args.n_rouge)]

    hypotheses = []
    for i in tqdm(range(args.n_rouge), desc="MDLM generate (ROUGE)"):
        prompt = test_prompted_ds[i]["prompt"]
        hyp = sample_mdlm(mdlm_model, tokenizer, keywords=prompt, device=DEVICE, n_steps=80, max_length=64)
        hypotheses.append(hyp)
    scores = compute_rouge(references, hypotheses)
    print(f"\nMDLM ROUGE")
    print(f"  ROUGE-1: {scores['rouge1']:.2f}")
    print(f"  ROUGE-2: {scores['rouge2']:.2f}")
    print(f"  ROUGE-L: {scores['rougeL']:.2f}")

    hyps_gpt2 = []
    for i in tqdm(range(args.n_rouge), desc="GPT-2 generate (ROUGE)"):
        prompt = test_prompted_ds[i]["prompt"]
        gen = generate_review_gpt2(prompt=prompt, tokenizer=gpt2_tokenizer, model=gpt2_model, max_new_tokens=100, do_sample=True, device=DEVICE)
        hyps_gpt2.append(gen[0])
    scores_gpt2 = compute_rouge(references, hyps_gpt2)
    print(f"\nGPT-2 ROUGE")
    print(f"  ROUGE-1: {scores_gpt2['rouge1']:.2f}")
    print(f"  ROUGE-2: {scores_gpt2['rouge2']:.2f}")
    print(f"  ROUGE-L: {scores_gpt2['rougeL']:.2f}")

    # --- BERTScore ---
    from bert_score import score as bert_score
    P_mdlm, R_mdlm, F_mdlm = bert_score(hypotheses, references, lang="en")
    P_gpt2, R_gpt2, F_gpt2 = bert_score(hyps_gpt2, references, lang="en")
    print("\nBERTScore F1 (higher = better semantic overlap):")
    print(f"  MDLM:  {F_mdlm.mean().item():.4f}")
    print(f"  GPT-2: {F_gpt2.mean().item():.4f}")

    # --- Sentiment accuracy (MDLM only) ---
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if DEVICE == "cuda" else -1
    )

    correct = 0
    total = 0
    N = args.n_sentiment

    for i in tqdm(range(N), desc="Evaluating MDLM sentiment"):
        example = test_prompted_ds[i]
        label = example["label"]
        keywords = example["prompt"]

        generated = sample_mdlm(
            mdlm_model,
            tokenizer,
            keywords=keywords,
            max_length=128,
            device=DEVICE
        )

        pred = sentiment_model(generated)[0]["label"]
        pred_label = 1 if pred == "POSITIVE" else 0

        if pred_label == label:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"\nMDLM Sentiment Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
