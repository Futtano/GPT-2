from typing import Any
from collections.abc import Mapping

import torch
import torch.nn as nn
from gpt_2.sattn import MultiHeadAttention

from gpt_2.config import ModelConfig, ensure_model_config

# Configuration of GPT-2 small model
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # vocabulary size for the BPE tokenizer
    "context_length": 1024,   # the maximum number of input tokens for the model
    "emb_dim": 768,           # the number of dimensions for the embedding of input tokens
    "n_heads": 12,            # the number of attention heads for the GPT-2 model
    "n_layers": 12,           # the number of transformer blocks for the GPT-2 model
    "drop_rate": 0.1,         # the dropout rate for the dropout layer
    "qkv_bias": False,        # whether to add or not a bias vector to linear layers of Q, K, V matrices
}

class LayerNorm(nn.Module):
  def __init__(self, emb_dim: int):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift

class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
    ))

class FeedForward(nn.Module):
  def __init__(self, emb_dim: int):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(emb_dim, 4*emb_dim),
        GELU(),
        nn.Linear(4*emb_dim, emb_dim),
    )

  def forward(self, x):
    return self.layers(x)

class TransformerBlock(nn.Module):
  def __init__(self, cfg: ModelConfig | Mapping[str, Any]):
    super().__init__()
    self.config = ensure_model_config(cfg)
    self.att = MultiHeadAttention(
        d_in=self.config.emb_dim,
        d_out=self.config.emb_dim,
        context_length=self.config.context_length,
        num_heads=self.config.n_heads,
        dropout=self.config.drop_rate,
        qkv_bias=self.config.qkv_bias,
    )
    self.ff = FeedForward(self.config.emb_dim)
    self.norm1 = LayerNorm(self.config.emb_dim)
    self.norm2 = LayerNorm(self.config.emb_dim)
    self.drop_shortcut = nn.Dropout(self.config.drop_rate)

  def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    return x

class GPTModel(nn.Module):
  def __init__(self, cfg: ModelConfig | Mapping[str, Any]):
    super().__init__()
    config = ensure_model_config(config=cfg)
    self.config = config
    self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.emb_dim)
    self.pos_emb = nn.Embedding(self.config.context_length, self.config.emb_dim)
    self.drop_emb = nn.Dropout(self.config.drop_rate)

    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(self.config)
        for _ in range(self.config.n_layers)]
    )

    self.final_norm = LayerNorm(self.config.emb_dim)
    self.out_head = nn.Linear(self.config.emb_dim, self.config.vocab_size, bias=False)

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(
        torch.arange(seq_len, device=in_idx.device)
    )
    x = tok_embeds + pos_embeds
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)

    return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(idx_cond)

    logits = logits[:, -1, :]
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    idx = torch.cat((idx_cond, idx_next), dim=1)

  return idx