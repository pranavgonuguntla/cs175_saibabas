# Discrete Diffusion for Keyword-Based Review Generation

CS 175 Project — Sai Babas

Explores text generation for Yelp reviews using **MDLM (Masked Diffusion Language Model)** conditioned on extracted keywords and sentiment, compared against a fine-tuned **GPT-2** baseline.

---

## Overview

Given a set of keywords (e.g. `"negative dry service korean bbq"`), the model generates a plausible Yelp review. Two approaches are compared:

- **MDLM** — a masked discrete diffusion transformer with classifier-free guidance (CFG)
- **GPT-2** — a fine-tuned autoregressive baseline using the same prompt format

Keywords are extracted from reviews using [KeyBERT](https://github.com/MaartenGr/KeyBERT) and combined with a sentiment label (`positive`/`negative`) from the [yelp_polarity](https://huggingface.co/datasets/yelp_polarity) dataset.

---

## Project Structure

```
├── config.py               # All shared constants (seed, device, hyperparameters)
├── train_mdlm.py           # Train the MDLM diffusion model
├── train_gpt2.py           # Fine-tune GPT-2 baseline
├── evaluate.py             # Full evaluation suite (perplexity, ROUGE, BERTScore, sentiment accuracy)
├── requirements.txt
│
├── data/
│   ├── preprocessing.py    # Text normalization and length clipping
│   ├── keywords.py         # KeyBERT/RAKE keyword extraction with disk cache
│   ├── prompt.py           # Prompt construction from keywords + sentiment label
│   └── dataset.py          # YelpPromptedDataset, MDLMYelpDataset, DataLoader collator
│
├── model/
│   ├── transformer.py      # AdaLN, MDLMTransformer (bidirectional transformer)
│   └── diffusion.py        # Masking schedule, training step, iterative sampling (CFG)
│
├── baseline/
│   ├── gpt2.py             # GPT-2 inference, GPT2YelpDataset, GPT2EvalDataset
│   └── sanity_check.py     # Zero-shot GPT-2 generation + sentiment conditioning test
│
└── evaluation/
    └── metrics.py          # ROUGE scoring utility
```

---

## Setup

```bash
pip install -r requirements.txt
```

NLTK data is downloaded automatically on first run.

---

## Usage

### Train MDLM

```bash
python train_mdlm.py
```

Trains for 10 epochs on up to 560k Yelp reviews. Saves model to `mdlm_model.pth`. After training, runs generation samples across guidance scales `[0.5, 1.0, ..., 10.0]`.

### Fine-tune GPT-2 baseline

```bash
python train_gpt2.py
```

Fine-tunes GPT-2 on 50k prompted Yelp examples. Saves to `gpt2_yelp_finetuned/`.

### GPT-2 zero-shot sanity check

```bash
python baseline/sanity_check.py
```

Runs pretrained (not fine-tuned) GPT-2 on test prompts to establish a zero-shot baseline.

### Evaluate

```bash
python evaluate.py --mdlm_path mdlm_model.pth --gpt2_path gpt2_yelp_finetuned
```

Reports for both models:
- **Pseudo-perplexity** (on 1,000 test examples)
- **ROUGE-1/2/L** (on 10 generated samples)
- **BERTScore F1**
- **Sentiment accuracy** — distilbert classifier on 200 MDLM generations

---

## Model Details

### MDLM (Masked Diffusion Language Model)

- Bidirectional transformer encoder (8 layers, 768 hidden, 12 heads)
- Conditioning via **AdaLN** (adaptive layer norm): keyword embeddings modulate hidden states
- **Cosine noise schedule**: masking probability `cos(((t + s)/(1 + s)) * π/2)²`
- **Classifier-free guidance** during sampling: `logits = logits_uncond + γ * (logits_cond - logits_uncond)`
- Iterative unmasking by confidence over 150 steps

### GPT-2 Baseline

- Pretrained `gpt2` (117M) fine-tuned on prompted reviews
- Prompt format: `"{sentiment} {keywords}\n\nReview: {text}"`
- CFG dropout (15%) applied during training so the model learns both conditional and unconditional generation

---

## Dataset

[`yelp_polarity`](https://huggingface.co/datasets/yelp_polarity) — 560k train / 38k test Yelp reviews with binary sentiment labels.

Keywords extracted per review using KeyBERT (`all-MiniLM-L6-v2`), cached to `cache/keyword_cache.json` to avoid re-extraction.
