"""Wrapper to make SMI-TED's encoder compatible with the ContinuationModel framework."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import get_paths

# Add SMI-TED inference code to path
SMI_TED_DIR = get_paths().smi_ted_dir
SMI_TED_CODE = SMI_TED_DIR / "smi-ted" / "inference" / "smi_ted_light"
sys.path.insert(0, str(SMI_TED_CODE))

import warnings
warnings.filterwarnings("ignore", message="invalid escape sequence")

# Patch torch.qr deprecation that breaks DataParallel with fast_transformers
import torch as _torch
if not hasattr(_torch, '_original_qr'):
    _torch._original_qr = _torch.qr
    def _patched_qr(input, some=True):
        mode = 'reduced' if some else 'complete'
        return _torch.linalg.qr(input, mode)
    _torch.qr = _patched_qr

from load import MoLEncoder, RotateEncoderBuilder, LangLayer


class SmiTedForMLM(nn.Module):
    """Wraps SMI-TED's MoLEncoder to match the RobertaForMaskedLM interface.

    Provides:
    - forward(input_ids, attention_mask, labels) -> .loss, .logits
    - .roberta-like encode via encode_hidden()
    - .config with hidden_size, num_attention_heads
    """

    def __init__(self, config: dict, n_vocab: int):
        super().__init__()
        self.encoder = MoLEncoder(config, n_vocab)
        self.lang_model = LangLayer(config["n_embd"], n_vocab)
        self.config = SimpleNamespace(
            hidden_size=config["n_embd"],
            num_attention_heads=config["n_head"],
            num_hidden_layers=config["n_layer"],
            vocab_size=n_vocab,
            max_position_embeddings=config["max_len"],
        )
        self._n_vocab = n_vocab
        self._max_len = config["max_len"]
        self._n_embd = config["n_embd"]

    def warmup_feature_maps(self, device=None):
        """Pre-initialize random feature maps to avoid lazy init crash in DataParallel."""
        if device is None:
            device = next(self.parameters()).device
        dummy = torch.randn(1, 4, self.encoder.config["n_head"],
                           self.encoder.config["n_embd"] // self.encoder.config["n_head"],
                           device=device)
        for layer in self.encoder.blocks.layers:
            attn = layer.attention
            if hasattr(attn, "inner_attention") and hasattr(attn.inner_attention, "feature_map"):
                attn.inner_attention.feature_map.new_feature_map(device)

    def encode_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input to hidden states (for view/translation encoding)."""
        x = self.encoder.tok_emb(input_ids)
        x = self.encoder.drop(x)
        from fast_transformers.masking import LengthMask
        x = self.encoder.blocks(x, length_mask=LengthMask(attention_mask.sum(-1), max_len=input_ids.shape[1]))
        return x

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True, **kwargs):
        """Forward pass matching RobertaForMaskedLM interface."""
        hidden = self.encode_hidden(input_ids, attention_mask)
        logits = self.lang_model(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self._n_vocab),
                labels.to(logits.device).view(-1),
                ignore_index=-100,
            )

        if return_dict:
            return SimpleNamespace(loss=loss, logits=logits, last_hidden_state=hidden)
        return (loss, logits)

    def resize_token_embeddings(self, new_size: int):
        """Resize embedding and LM head for vocab extension."""
        old_size = self.encoder.tok_emb.num_embeddings
        if new_size == old_size:
            return
        old_emb = self.encoder.tok_emb
        new_emb = nn.Embedding(new_size, self._n_embd)
        new_emb.weight.data[:old_size] = old_emb.weight.data
        self.encoder.tok_emb = new_emb

        old_head = self.lang_model.head
        new_head = nn.Linear(self._n_embd, new_size, bias=False)
        new_head.weight.data[:old_size] = old_head.weight.data
        self.lang_model.head = new_head

        self.config.vocab_size = new_size
        self._n_vocab = new_size


def build_smi_ted_scratch() -> SmiTedForMLM:
    """Build SMI-TED model with random weights (no pretrained checkpoint)."""
    ckpt_path = SMI_TED_DIR / "smi-ted-Light_40.pt"
    vocab_path = SMI_TED_DIR / "bert_vocab_curated.txt"
    with open(vocab_path) as f:
        n_vocab = len(f.readlines())
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint["hparams"]
    return SmiTedForMLM(config, n_vocab)


def load_smi_ted_for_mlm() -> SmiTedForMLM:
    """Load pretrained SMI-TED encoder weights into SmiTedForMLM."""
    ckpt_path = SMI_TED_DIR / "smi-ted-Light_40.pt"
    vocab_path = SMI_TED_DIR / "bert_vocab_curated.txt"

    # Count vocab
    with open(vocab_path) as f:
        n_vocab = len(f.readlines())

    # Load checkpoint to get config
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint["hparams"]

    # Build model
    model = SmiTedForMLM(config, n_vocab)

    # Load encoder weights
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if isinstance(state_dict, list):
            # state_dict[0] = encoder, state_dict[1] = decoder
            encoder_sd = state_dict[0]
            model.encoder.load_state_dict(encoder_sd, strict=False)
            # Load lang_model from encoder's lang_model
            lang_keys = {k: v for k, v in encoder_sd.items() if k.startswith("lang_model.")}
            if lang_keys:
                model.lang_model.load_state_dict(
                    {k.replace("lang_model.", ""): v for k, v in lang_keys.items()},
                    strict=False,
                )
        else:
            # Try loading full state dict
            encoder_keys = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
            if encoder_keys:
                model.encoder.load_state_dict(
                    {k.replace("encoder.", ""): v for k, v in encoder_keys.items()},
                    strict=False,
                )

    return model
