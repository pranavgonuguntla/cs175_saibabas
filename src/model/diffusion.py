"""
model/diffusion.py — Masked diffusion forward process, training step, and sampler.

Implements the core MDLM (Masked Diffusion Language Model) logic:

  get_mask_prob(t, schedule)
    Maps a continuous noise level t ∈ [0, 1] to a masking probability using a
    cosine schedule (default) or linear schedule. At t=0 the sequence is
    fully unmasked; at t=1 it is fully masked.

  create_masked_input(clean_tokens, mask_token_id, mask_prob, attention_mask)
    Applies token-level masking stochastically given a per-example mask_prob.
    Only real (non-pad) positions are eligible for masking when attention_mask
    is provided.

  train_step(model, batch, optimizer, mask_token_id, schedule, device)
    One gradient update. Samples a random noise level t per example, masks
    the input accordingly, runs the model, computes cross-entropy loss only
    on the masked positions, clips gradients (max_norm=1.0), and steps the
    optimizer. Returns (loss, accuracy) scalars.

  sample_mdlm(model, tokenizer, keywords, n_steps, guidance_scale, ...)
    Iterative confidence-based unmasking (masked diffusion sampling):
      1. Start from a fully masked sequence.
      2. At each step, run the model to get per-position token probabilities.
      3. If guidance_scale > 1.0, apply classifier-free guidance:
             logits = logits_uncond + γ * (logits_cond - logits_uncond)
      4. Among currently masked positions, unmask the top-confidence ones
         (fraction = 1 / remaining_steps), filling them with sampled tokens.
      5. Repeat for n_steps (default 150); return the decoded string.
"""
import torch
import torch.nn.functional as F
import numpy as np


def get_mask_prob(t, schedule='cosine'):
    if schedule == 'cosine':
        s = 0.008
        return np.cos(((t + s) / (1 + s)) * np.pi / 2) ** 2
    else:
        return t


def create_masked_input(clean_tokens, mask_token_id, mask_prob, attention_mask=None):
    batch_size, seq_len = clean_tokens.shape
    device = clean_tokens.device
    random_mask = torch.rand(clean_tokens.shape, device=device)
    if isinstance(mask_prob, torch.Tensor) and mask_prob.dim() > 0:
        mask_prob = mask_prob.view(batch_size, 1)
    mask_indices = random_mask < mask_prob
    if attention_mask is not None:
        mask_indices = mask_indices & (attention_mask.bool())
    noisy_input = clean_tokens.clone()
    noisy_input[mask_indices] = mask_token_id
    return noisy_input, mask_indices


def train_step(model, batch, optimizer, mask_token_id, schedule='cosine', device='cuda'):
    model.train()
    clean_tokens = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    keyword_ids = batch["keyword_ids"].to(device)
    batch_size, seq_len = clean_tokens.shape

    t = torch.rand(batch_size, 1, device=device)
    mask_probs = torch.tensor(
        [get_mask_prob(t_i.item(), schedule) for t_i in t], device=device
    ).view(batch_size, 1)

    noisy_input, mask_indices = create_masked_input(
        clean_tokens, mask_token_id, mask_probs, attention_mask
    )

    logits = model(noisy_input, condition_tokens=keyword_ids)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), clean_tokens.view(-1), reduction='none'
    )
    loss = loss.view(batch_size, seq_len)
    masked_loss = (loss * mask_indices.float()).sum() / (mask_indices.sum() + 1e-8)

    optimizer.zero_grad()
    masked_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        correct = (predictions == clean_tokens) & mask_indices
        accuracy = correct.sum().float() / (mask_indices.sum() + 1e-8)

    return masked_loss.item(), accuracy.item()


@torch.no_grad()
def sample_mdlm(model, tokenizer, keywords=None, n_steps=150, guidance_scale=3.0,
                temperature=1.0, max_length=128, device='cuda'):

    model.eval()
    vocab_size = model.vocab_size
    mask_token_id = vocab_size - 1

    if keywords is None or keywords == "<NULL>":
        keyword_ids = torch.full((1, 32), tokenizer.pad_token_id, dtype=torch.long, device=device)
        use_cfg = False
    else:
        keyword_encoding = tokenizer(
            keywords,
            truncation=True,
            max_length=32,
            padding="max_length",
            return_tensors="pt"
        )
        keyword_ids = keyword_encoding["input_ids"].to(device)
        keyword_ids = torch.clamp(keyword_ids, max=vocab_size - 1)
        use_cfg = guidance_scale > 1.0

    mask_token_id = model.vocab_size - 1
    x = torch.full((1, max_length), mask_token_id, dtype=torch.long, device=device)
    null_keyword_ids = torch.full((1, 32), tokenizer.pad_token_id, dtype=torch.long, device=device)

    for step in range(n_steps):
        x = torch.clamp(x, max=vocab_size - 1)

        if use_cfg:
            logits_cond = model(x, condition_tokens=keyword_ids)
            logits_uncond = model(x, condition_tokens=null_keyword_ids)
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        else:
            logits = model(x, condition_tokens=keyword_ids)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        sampled_tokens = torch.multinomial(probs[0], num_samples=1).squeeze(-1)

        mask_positions = (x[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) > 0:
            n_unmask = max(1, int(len(mask_positions) * (1.0 / (n_steps - step))))
            masked_probs = probs[0, mask_positions]
            max_probs, _ = masked_probs.max(dim=-1)

            _, confident_indices = torch.topk(
                max_probs,
                min(n_unmask, len(mask_positions))
            )

            positions_to_unmask = mask_positions[confident_indices]
            x[0, positions_to_unmask] = sampled_tokens[positions_to_unmask]

    return tokenizer.decode(x[0].cpu(), skip_special_tokens=True)
