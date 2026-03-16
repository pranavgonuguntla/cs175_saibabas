"""
evaluate.py — Full evaluation suite comparing MDLM and fine-tuned GPT-2.

Loads both trained models and runs a comprehensive side-by-side comparison
across the yelp_polarity test set. All metrics are printed in a single table.

Metrics computed for both models:

  Perplexity (masked diffusion loss)
    Uses the same masked-input / masked-position loss as the training loop,
    so numbers are directly comparable to training validation loss. Evaluated
    on a 1,000-example subset of the test set.

  ROUGE-1 / ROUGE-2 / ROUGE-L
    N-gram overlap between generated and reference reviews (n_rouge examples,
    default 10). Generation uses n_steps=150, max_length=128 to match training
    defaults. Seeded before each loop for reproducibility.

  BERTScore F1
    Semantic similarity between generated and reference reviews using a
    pretrained BERT model. Complements ROUGE by capturing paraphrase overlap.

  Sentiment Accuracy
    Fraction of generated reviews whose predicted sentiment (DistilBERT
    fine-tuned on SST-2) matches the conditioning label. Measures how well
    each model follows its conditioning signal.

  Keyword Recall
    Fraction of conditioning keywords that appear verbatim in the generated
    text. Directly measures adherence to the keyword conditioning.

  Grammar Acceptability (CoLA)
    Fraction of generated reviews classified as grammatically acceptable by
    the textattack/bert-base-uncased-CoLA model (LABEL_1 = acceptable).

  Distinct-1 / Distinct-2
    Ratio of unique unigrams / bigrams over all generated tokens. Higher means
    more diverse output; lower means more repetitive generation.

  Generation time
    Wall-clock seconds for the full generation batch and per sample.

Run from the project root:
    python src/evaluate.py --mdlm_path mdlm_model.pth --gpt2_path gpt2_yelp_finetuned
"""
import argparse
import math
import time
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
    """Masked diffusion loss — identical to the training validation loop."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="MDLM perplexity"):
            clean_tokens   = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            keyword_ids    = batch["keyword_ids"].to(device)
            batch_size, seq_len = clean_tokens.shape

            t = torch.rand(batch_size, 1, device=device)
            mask_probs = torch.tensor(
                [get_mask_prob(t_i.item(), 'cosine') for t_i in t], device=device
            ).view(batch_size, 1)

            noisy_input, mask_indices = create_masked_input(
                clean_tokens, tokenizer.mask_token_id, mask_probs, attention_mask
            )

            logits = model(noisy_input, condition_tokens=keyword_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), clean_tokens.view(-1), reduction='none'
            ).view(batch_size, seq_len)
            masked_loss = (loss * mask_indices.float()).sum() / (mask_indices.sum() + 1e-8)

            total_loss += masked_loss.item()
            total_batches += 1

    mdlm_loss = total_loss / max(total_batches, 1)
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

    torch.manual_seed(SEED)
    hypotheses = []
    for i in tqdm(range(args.n_rouge), desc="MDLM generate (ROUGE)"):
        prompt = test_prompted_ds[i]["prompt"]
        hyp = sample_mdlm(mdlm_model, tokenizer, keywords=prompt, device=DEVICE, n_steps=150, max_length=128)
        hypotheses.append(hyp)
    scores = compute_rouge(references, hypotheses)
    print(f"\nMDLM ROUGE")
    print(f"  ROUGE-1: {scores['rouge1']:.2f}")
    print(f"  ROUGE-2: {scores['rouge2']:.2f}")
    print(f"  ROUGE-L: {scores['rougeL']:.2f}")

    torch.manual_seed(SEED)
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

    # --- Sentiment + keyword recall + grammar (both models) ---
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if DEVICE == "cuda" else -1,
    )
    grammar_model = pipeline(
        "text-classification",
        model="textattack/bert-base-uncased-CoLA",
        device=0 if DEVICE == "cuda" else -1,
    )

    N = args.n_sentiment

    def _eval_generations(generated_texts, test_ds, N, desc):
        """Shared scorer: sentiment accuracy, keyword recall, distinct-N, grammar."""
        sent_correct = 0
        kw_recall_total = 0.0
        all_tokens = []
        grammar_acceptable = 0

        for i, generated in enumerate(tqdm(generated_texts, desc=desc)):
            example = test_ds[i]
            label = example["label"]
            prompt_kws = example["prompt"].lower().split()

            # Sentiment
            pred = sentiment_model(generated[:512])[0]["label"]
            if (1 if pred == "POSITIVE" else 0) == label:
                sent_correct += 1

            # Keyword recall
            gen_lower = generated.lower()
            hits = sum(1 for kw in prompt_kws if kw in gen_lower)
            kw_recall_total += hits / max(len(prompt_kws), 1)

            # Grammar (CoLA) — model outputs LABEL_0=unacceptable, LABEL_1=acceptable
            result = grammar_model(generated[:512])[0]
            if result["label"] == "LABEL_1":
                grammar_acceptable += 1

            # Distinct tokens
            all_tokens.extend(generated.lower().split())

        sent_acc = sent_correct / N
        kw_recall = kw_recall_total / N
        grammar_rate = grammar_acceptable / N

        unigrams = all_tokens
        bigrams = list(zip(all_tokens, all_tokens[1:]))
        dist1 = len(set(unigrams)) / max(len(unigrams), 1)
        dist2 = len(set(bigrams)) / max(len(bigrams), 1)

        return sent_acc, kw_recall, grammar_rate, dist1, dist2

    # Generate MDLM outputs
    torch.manual_seed(SEED)
    mdlm_gens = []
    t0 = time.perf_counter()
    for i in tqdm(range(N), desc="MDLM generate (sentiment/grammar)"):
        example = test_prompted_ds[i]
        gen = sample_mdlm(mdlm_model, tokenizer, keywords=example["prompt"], max_length=128, device=DEVICE)
        mdlm_gens.append(gen)
    mdlm_gen_time = time.perf_counter() - t0

    # Generate GPT-2 outputs
    torch.manual_seed(SEED)
    gpt2_gens = []
    t0 = time.perf_counter()
    for i in tqdm(range(N), desc="GPT-2 generate (sentiment/grammar)"):
        example = test_prompted_ds[i]
        gen = generate_review_gpt2(prompt=example["prompt"], tokenizer=gpt2_tokenizer, model=gpt2_model,
                                   max_new_tokens=100, do_sample=True, device=DEVICE)
        gpt2_gens.append(gen[0])
    gpt2_gen_time = time.perf_counter() - t0

    mdlm_sent, mdlm_kw, mdlm_gram, mdlm_d1, mdlm_d2 = _eval_generations(mdlm_gens, test_prompted_ds, N, "Scoring MDLM")
    gpt2_sent, gpt2_kw, gpt2_gram, gpt2_d1, gpt2_d2 = _eval_generations(gpt2_gens, test_prompted_ds, N, "Scoring GPT-2")

    print(f"\n{'Metric':<28} {'MDLM':>10} {'GPT-2':>10}")
    print("-" * 50)
    print(f"{'Sentiment Accuracy':<28} {mdlm_sent:>10.4f} {gpt2_sent:>10.4f}")
    print(f"{'Keyword Recall':<28} {mdlm_kw:>10.4f} {gpt2_kw:>10.4f}")
    print(f"{'Grammar Acceptability (CoLA)':<28} {mdlm_gram:>10.4f} {gpt2_gram:>10.4f}")
    print(f"{'Distinct-1':<28} {mdlm_d1:>10.4f} {gpt2_d1:>10.4f}")
    print(f"{'Distinct-2':<28} {mdlm_d2:>10.4f} {gpt2_d2:>10.4f}")
    print(f"{'Generation Time (s)':<28} {mdlm_gen_time:>10.2f} {gpt2_gen_time:>10.2f}")
    print(f"{'Time per sample (s)':<28} {mdlm_gen_time/N:>10.2f} {gpt2_gen_time/N:>10.2f}")


if __name__ == "__main__":
    main()
