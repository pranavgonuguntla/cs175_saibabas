"""
plot_guidance.py — Analyse and plot MDLM quality metrics across guidance scales.

Generates n_samples reviews at each of the 10 guidance scales
[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0] and measures five
quality dimensions, then saves a 6-panel figure to guidance_scale_analysis.png.

Metrics measured at each scale:

  Sentiment Accuracy
    Fraction of generated reviews whose DistilBERT-predicted sentiment matches
    the conditioning label. Shows whether the guidance scale is strong enough
    to actually steer the content.

  Grammar Acceptability (CoLA)
    Fraction of reviews classified as grammatically acceptable by the
    textattack/bert-base-uncased-CoLA model. High guidance scales can degrade
    fluency; this tracks the trade-off.

  Distinct-1 / Distinct-2
    Unique unigram / bigram ratios across all generated text. Low values mean
    the model is collapsing to repetitive patterns (common at extreme scales).

  Avg Generation Time (s/sample)
    Wall-clock time per generated review. Constant across scales (same
    n_steps=150) but useful for comparing against GPT-2 in evaluate.py.

  Repetitiveness (1 − Distinct-2)
    Inverted Distinct-2 plotted as a more intuitive "how repetitive" panel.

The RNG is re-seeded to SEED before each guidance scale so differences between
scales reflect the guidance effect, not random variation.

Run from the project root:
    python src/plot_guidance.py --mdlm_path mdlm_model.pth --n_samples 200
"""
import argparse
import time

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm

from config import DEVICE, SEED, YELP_DATASET_NAME, INCLUDE_SENTIMENT, MAX_KEYWORDS, KEYWORD_TOP_K, MAX_TEST_EXAMPLES
from datasets import load_dataset
from data.dataset import build_splits_for_sedd
from model.transformer import MDLMTransformer
from model.diffusion import sample_mdlm

GUIDANCE_SCALES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]


def load_mdlm(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"mask_token": "[MASK]", "pad_token": "<PAD>"})
    vocab_size = len(tokenizer)
    model = MDLMTransformer(
        vocab_size=vocab_size, d_model=768, nhead=12,
        num_layers=8, dim_feedforward=3072, max_seq_len=128,
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval().to(device)
    return model, tokenizer


def distinct_n(texts, n):
    all_ngrams, unique_ngrams = [], set()
    for text in texts:
        tokens = text.lower().split()
        ngrams = list(zip(*[tokens[i:] for i in range(n)]))
        all_ngrams.extend(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams) / max(len(all_ngrams), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm_path", default="mdlm_model.pth")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples per guidance scale")
    args = parser.parse_args()

    print("Loading model and data...")
    model, tokenizer = load_mdlm(args.mdlm_path, DEVICE)

    raw_datasets = load_dataset(YELP_DATASET_NAME)
    _, test_ds = build_splits_for_sedd(
        raw_datasets=raw_datasets,
        max_test_examples=MAX_TEST_EXAMPLES,
        cfg_drop_prob=0.0,
        include_sentiment=INCLUDE_SENTIMENT,
        max_keywords=MAX_KEYWORDS,
        keyword_top_k=KEYWORD_TOP_K,
        seed=SEED,
    )

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if DEVICE == "cuda" else -1,
        truncation=True, max_length=512,
    )
    grammar_pipe = pipeline(
        "text-classification",
        model="textattack/bert-base-uncased-CoLA",
        device=0 if DEVICE == "cuda" else -1,
        truncation=True, max_length=512,
    )

    prompts = [test_ds[i]["prompt"] for i in range(args.n_samples)]
    labels  = [test_ds[i]["label"]  for i in range(args.n_samples)]

    metrics = {s: {} for s in GUIDANCE_SCALES}

    for scale in GUIDANCE_SCALES:
        print(f"\n--- guidance_scale={scale} ---")
        torch.manual_seed(SEED)

        texts = []
        t0 = time.perf_counter()
        for prompt in tqdm(prompts, desc="generating"):
            text = sample_mdlm(model, tokenizer, keywords=prompt,
                               n_steps=150, guidance_scale=scale,
                               max_length=128, device=DEVICE)
            texts.append(text)
        gen_time = time.perf_counter() - t0

        # Sentiment accuracy
        sent_correct = sum(
            (1 if sentiment_pipe(t[:512])[0]["label"] == "POSITIVE" else 0) == lbl
            for t, lbl in zip(texts, labels)
        )

        # Grammar acceptability (LABEL_1 = acceptable)
        gram_ok = sum(
            1 for t in texts if grammar_pipe(t[:512])[0]["label"] == "LABEL_1"
        )

        metrics[scale]["sentiment_acc"]   = sent_correct / args.n_samples
        metrics[scale]["grammar_acc"]     = gram_ok / args.n_samples
        metrics[scale]["distinct1"]       = distinct_n(texts, 1)
        metrics[scale]["distinct2"]       = distinct_n(texts, 2)
        metrics[scale]["gen_time"]        = gen_time / args.n_samples  # per sample

    # --- Plot ---
    scales = GUIDANCE_SCALES
    sent_vals   = [metrics[s]["sentiment_acc"] for s in scales]
    gram_vals   = [metrics[s]["grammar_acc"]   for s in scales]
    dist1_vals  = [metrics[s]["distinct1"]     for s in scales]
    dist2_vals  = [metrics[s]["distinct2"]     for s in scales]
    time_vals   = [metrics[s]["gen_time"]      for s in scales]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("MDLM — Metrics Across Guidance Scales", fontsize=14)

    def _plot(ax, y, title, ylabel, color):
        ax.plot(scales, y, marker="o", color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Guidance Scale")
        ax.set_ylabel(ylabel)
        ax.set_xticks(scales)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], sent_vals,  "Sentiment Accuracy",         "Accuracy",    "steelblue")
    _plot(axes[0, 1], gram_vals,  "Grammar Acceptability (CoLA)", "Rate",      "seagreen")
    _plot(axes[0, 2], dist1_vals, "Distinct-1 (Diversity)",     "Distinct-1",  "darkorange")
    _plot(axes[1, 0], dist2_vals, "Distinct-2 (Diversity)",     "Distinct-2",  "mediumpurple")
    _plot(axes[1, 1], time_vals,  "Avg Generation Time",        "Seconds/sample", "firebrick")

    # Combined repetitiveness: lower distinct = more repetitive, invert for clarity
    rep_vals = [1 - d for d in dist2_vals]
    _plot(axes[1, 2], rep_vals, "Repetitiveness (1 - Distinct-2)", "Score", "saddlebrown")

    plt.tight_layout()
    out_path = "guidance_scale_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
