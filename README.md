# Discrete Diffusion for Keyword-Based Review Generation

CS 175 Project — Sai Babas

Explores text generation for Yelp reviews using **MDLM (Masked Diffusion Language Model)** conditioned on extracted keywords and sentiment, compared against a fine-tuned **GPT-2** baseline.

Trained MDLM model: https://huggingface.co/xenosaac/Sai_BaBa_MDLM/tree/main

The libraries we used are:
torch
transformers
datasets
numpy
pandas
nltk
tqdm
scikit-learn
keybert
sentence-transformers
rouge-score
bert-score
rake-nltk
matplotlib

---

## Overview

Given a set of keywords (e.g. `"negative dry service korean bbq"`), the model generates a plausible Yelp review. Two approaches are compared:

- **MDLM** — a masked discrete diffusion transformer with classifier-free guidance (CFG)
- **GPT-2** — a fine-tuned autoregressive baseline using the same prompt format

Keywords are extracted from reviews using [KeyBERT](https://github.com/MaartenGr/KeyBERT) and combined with a sentiment label (`positive`/`negative`) from the [yelp_polarity](https://huggingface.co/datasets/yelp_polarity) dataset.

This model is based on this github - https://github.com/kuleshov-group/mdlm

---

## Project Structure

```
├── src/
│   ├── config.py               # All shared constants (seed, device, hyperparameters)
│   ├── train_mdlm.py           # Train the MDLM diffusion model
│   ├── train_gpt2.py           # Fine-tune GPT-2 baseline
│   ├── evaluate.py             # Full evaluation suite (perplexity, ROUGE, BERTScore, sentiment, grammar, diversity)
│   ├── generate.py             # Interactive generation across all guidance scales
│   ├── plot_guidance.py        # Plot metrics across guidance scales (sentiment, grammar, distinct-N, timing)
│   │
│   ├── data/
│   │   ├── preprocessing.py    # Text normalization and length clipping
│   │   ├── keywords.py         # KeyBERT/RAKE keyword extraction with disk cache
│   │   ├── prompt.py           # Prompt construction from keywords + sentiment label
│   │   └── dataset.py          # YelpPromptedDataset, MDLMYelpDataset, DataLoader collator
│   │
│   ├── model/
│   │   ├── transformer.py      # AdaLN, MDLMTransformer (bidirectional transformer)
│   │   └── diffusion.py        # Masking schedule, training step, iterative sampling (CFG)
│   │
│   ├── baseline/
│   │   ├── gpt2.py             # GPT-2 inference, GPT2YelpDataset, GPT2EvalDataset
│   │   └── sanity_check.py     # Zero-shot GPT-2 generation + sentiment conditioning test
│   │
│   └── evaluation/
│       └── metrics.py          # ROUGE scoring utility
│
├── cache/                      # KeyBERT keyword cache (auto-generated)
├── mdlm_model.pth              # Saved MDLM weights
├── mdlm_tokenizer/             # Saved tokenizer (written by train_mdlm.py)
├── gpt2_yelp_finetuned/        # Saved fine-tuned GPT-2
├── guidance_scale_analysis.png # Output from plot_guidance.py
├── project.ipynb               # Development notebook
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

NLTK data is downloaded automatically on first run. All scripts should be run from the **project root** (not from inside `src/`).

---

## Usage

### Train MDLM

```bash
python src/train_mdlm.py
```

Trains for 10 epochs on up to 560k Yelp reviews. Saves weights to `mdlm_model.pth` and tokenizer to `mdlm_tokenizer/`. After training, runs generation samples across guidance scales `[0.5, 1.0, ..., 10.0]`.

### Fine-tune GPT-2 baseline

```bash
python src/train_gpt2.py
```

Fine-tunes GPT-2 on 50k prompted Yelp examples. Saves to `gpt2_yelp_finetuned/`.

### GPT-2 zero-shot sanity check

```bash
python src/baseline/sanity_check.py
```

Runs pretrained (not fine-tuned) GPT-2 on test prompts to establish a zero-shot baseline.

### Evaluate

```bash
python src/evaluate.py --mdlm_path mdlm_model.pth --gpt2_path gpt2_yelp_finetuned
```

Reports the following for both models side-by-side:

| Metric | Description |
|---|---|
| Pseudo-perplexity | Masked diffusion loss on 1,000 test examples (matches training validation) |
| ROUGE-1/2/L | N-gram overlap with reference reviews (150-step, 128-token generation) |
| BERTScore F1 | Semantic similarity to reference reviews |
| Sentiment accuracy | DistilBERT classifier on generated text vs. conditioning label |
| Keyword recall | Fraction of conditioning keywords that appear in generated text |
| Grammar acceptability | CoLA classifier (`textattack/bert-base-uncased-CoLA`) |
| Distinct-1 / Distinct-2 | Unique unigram/bigram ratio — measures output diversity |
| Generation time | Total and per-sample wall-clock time |

### Interactive generation

```bash
python src/generate.py --mdlm_path mdlm_model.pth
```

Prompts for keywords in a loop and generates reviews at all 10 guidance scales. Type `quit` to exit.

### Guidance scale analysis plots

```bash
python src/plot_guidance.py --mdlm_path mdlm_model.pth --n_samples 200
```

Generates text at each guidance scale and produces `guidance_scale_analysis.png` with 6 subplots: sentiment accuracy, grammar acceptability, Distinct-1, Distinct-2, generation time, and repetitiveness.

---

## Model Details

### MDLM (Masked Diffusion Language Model)

- Bidirectional transformer encoder (8 layers, 768 hidden, 12 heads, ~86M params)
- Conditioning via **AdaLN** (adaptive layer norm): keyword embeddings modulate hidden states before and after the transformer stack
- **Cosine noise schedule**: masking probability `cos(((t + s)/(1 + s)) * π/2)²`
- **Classifier-free guidance** (CFG) during sampling: `logits = logits_uncond + γ * (logits_cond - logits_uncond)`
- Iterative confidence-based unmasking over 150 steps
- Tokenizer: GPT-2 BPE + `[MASK]` and `<PAD>` special tokens (vocab size 50,259)

### GPT-2 Baseline

- Pretrained `gpt2` (117M) fine-tuned on prompted reviews
- Prompt format: `"{sentiment} {keywords}\n\nReview: {text}"`
- CFG dropout (15%) applied during training so the model learns both conditional and unconditional generation

---

## Dataset

[`yelp_polarity`](https://huggingface.co/datasets/yelp_polarity) — 560k train / 38k test Yelp reviews with binary sentiment labels.

Keywords extracted per review using KeyBERT (`all-MiniLM-L6-v2`), cached to `cache/keyword_cache.json` to avoid re-extraction at training and evaluation time.

---

