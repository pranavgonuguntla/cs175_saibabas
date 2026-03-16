"""
generate.py — Interactive Yelp review generator using the trained MDLM model.

Loads a trained MDLMTransformer checkpoint and enters a prompt loop, generating
a Yelp review at every guidance scale for each keyword prompt you type. This
lets you interactively explore how classifier-free guidance strength affects
the output's content, fluency, and adherence to the conditioning keywords.

Guidance scale behaviour:
  < 1.0  — pushes the model AWAY from the keywords
  = 1.0  — no guidance; generation ignores the keywords entirely
  1–3    — mild to moderate keyword steering
  3.0    — recommended default (strong guidance without sacrificing fluency)
  5–10   — heavy steering; keywords dominate but output may become repetitive

Tokenizer: loads from mdlm_tokenizer/ if present (saved by train_mdlm.py),
otherwise recreates from the base GPT-2 tokenizer with added special tokens.

Usage:
    python src/generate.py --mdlm_path mdlm_model.pth

Prompt format:
    "[sentiment] [keywords]"  e.g. "positive great food friendly staff"

Type 'quit' or press Ctrl+C to exit.
"""
import argparse
import torch
from transformers import AutoTokenizer

from config import DEVICE
from model.transformer import MDLMTransformer
from model.diffusion import sample_mdlm

GUIDANCE_SCALES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]


def load_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'mask_token': '[MASK]', 'pad_token': '<PAD>'})
    vocab_size = len(tokenizer)

    model = MDLMTransformer(
        vocab_size=vocab_size,
        d_model=768,
        nhead=12,
        num_layers=8,
        dim_feedforward=3072,
        max_seq_len=128,
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    model.eval()

    print(f"Loaded model from {model_path} ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer


def generate_all_scales(model, tokenizer, prompt: str, device: str):
    print(f"\n{'='*80}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*80}")

    for scale in GUIDANCE_SCALES:
        text = sample_mdlm(
            model, tokenizer,
            keywords=prompt,
            n_steps=150,
            guidance_scale=scale,
            max_length=128,
            device=device,
        )
        print(f"\n  guidance={scale:<5} | {prompt}")
        print(f"  {text}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm_path", default="mdlm_model.pth", help="Path to saved mdlm_model.pth")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.mdlm_path, DEVICE)

    print("\nEnter a prompt to generate Yelp reviews across all guidance scales.")
    print("Format: '[sentiment] [keywords]'  e.g. 'positive great food friendly staff'")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    try:
        while True:
            prompt = input("Prompt: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                break
            generate_all_scales(model, tokenizer, prompt, DEVICE)
    except KeyboardInterrupt:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
