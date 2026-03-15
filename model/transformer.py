import torch
import torch.nn as nn
import math


class AdaLN(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, d_model * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        x = self.norm(x)
        scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return x * (1 + scale) + shift


class MDLMTransformer(nn.Module):
    def __init__(self, vocab_size=50259, d_model=768, nhead=8, num_layers=8,
                 dim_feedforward=3072, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.cond_proj = nn.Linear(d_model, d_model)
        self.adaln = AdaLN(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.adaln.proj:
                    continue
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, condition_tokens=None):
        batch_size, seq_len = x.shape
        device = x.device

        token_emb = self.token_embedding(x)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        h = token_emb + pos_emb

        if condition_tokens is not None:
            cond_emb = self.token_embedding(condition_tokens)
            cond = self.cond_proj(cond_emb.mean(dim=1))
            h = self.adaln(h, cond)

        h = self.transformer(h)

        if condition_tokens is not None:
            h = self.adaln(h, cond)

        logits = self.output_layer(h)
        return logits
