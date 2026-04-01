from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExperimentalBackboneConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    layer_norm_eps: float = 1e-12
    position_embedding_type: str = "absolute"
    attention_variant: str = "mha"
    num_key_value_heads: int = 4


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class ExperimentalSelfAttention(nn.Module):
    def __init__(self, config: ExperimentalBackboneConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads if config.attention_variant == "gqa" else config.num_attention_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.head_dim * config.num_attention_heads != config.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.attention_variant = config.attention_variant
        self.position_embedding_type = config.position_embedding_type
        self.dropout = config.attention_probs_dropout_prob
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        if self.position_embedding_type == "rope":
            inv_freq = 1.0 / (
                10000
                ** (
                    torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                    / float(self.head_dim)
                )
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
        k = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)
        return q, k

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.position_embedding_type == "rope":
            q, k = self._apply_rope(q, k)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        attn_bias = (1.0 - attention_mask[:, None, None, :].float()) * -1e9
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class ExperimentalEncoderLayer(nn.Module):
    def __init__(self, config: ExperimentalBackboneConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = ExperimentalSelfAttention(config)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout1(self.attn(self.norm1(hidden_states), attention_mask))
        hidden_states = hidden_states + self.dropout2(self.ff(self.norm2(hidden_states)))
        return hidden_states


class ExperimentalEncoderForMLM(nn.Module):
    def __init__(self, config: ExperimentalBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if config.position_embedding_type == "absolute"
            else None
        )
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([ExperimentalEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_decoder.weight = self.token_embeddings.weight
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def encode_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.token_embeddings(input_ids)
        if self.position_embeddings is not None:
            positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            hidden_states = hidden_states + self.position_embeddings(positions)
        hidden_states = self.embedding_dropout(self.embedding_norm(hidden_states))
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.final_norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        hidden_states = self.encode_hidden(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_decoder(self.lm_norm(F.gelu(self.lm_dense(hidden_states)))) + self.lm_bias
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        if return_dict:
            return SimpleNamespace(loss=loss, logits=logits, last_hidden_state=hidden_states)
        return SimpleNamespace(loss=loss, logits=logits, last_hidden_state=hidden_states)
