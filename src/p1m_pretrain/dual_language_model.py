from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_tokenizer import DualTokenizerBundle, PSELFIES_LANGUAGE_ID, PSMILES_LANGUAGE_ID
from .experimental_backbone import ExperimentalBackboneConfig, ExperimentalEncoderLayer


@dataclass
class DualLanguageModelConfig:
    smiles_vocab_size: int
    selfies_vocab_size: int
    smiles_pad_id: int
    smiles_cls_id: int
    smiles_sep_id: int
    smiles_mask_id: int
    selfies_pad_id: int
    selfies_cls_id: int
    selfies_sep_id: int
    selfies_mask_id: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    local_encoder_layers: int = 1
    local_mlm_decoder_layers: int = 1
    local_translation_decoder_layers: int = 1


def build_dual_language_config(bundle: DualTokenizerBundle, scratch_variant: str = "base") -> DualLanguageModelConfig:
    presets = {
        "base": dict(num_hidden_layers=6, hidden_size=768, num_attention_heads=12, intermediate_size=3072),
        "deep": dict(num_hidden_layers=8, hidden_size=768, num_attention_heads=12, intermediate_size=3072),
        "small": dict(num_hidden_layers=4, hidden_size=512, num_attention_heads=8, intermediate_size=2048),
        "tiny": dict(num_hidden_layers=3, hidden_size=384, num_attention_heads=6, intermediate_size=1536),
    }
    if scratch_variant not in presets:
        raise ValueError(f"Unsupported scratch_variant: {scratch_variant}")
    preset = presets[scratch_variant]
    return DualLanguageModelConfig(
        smiles_vocab_size=len(bundle.psmiles_tokenizer),
        selfies_vocab_size=len(bundle.pselfies_tokenizer),
        smiles_pad_id=bundle.psmiles_tokenizer.pad_token_id,
        smiles_cls_id=bundle.psmiles_tokenizer.cls_token_id,
        smiles_sep_id=bundle.psmiles_tokenizer.sep_token_id,
        smiles_mask_id=bundle.psmiles_tokenizer.mask_token_id,
        selfies_pad_id=bundle.pselfies_tokenizer.pad_token_id,
        selfies_cls_id=bundle.pselfies_tokenizer.cls_token_id,
        selfies_sep_id=bundle.pselfies_tokenizer.sep_token_id,
        selfies_mask_id=bundle.pselfies_tokenizer.mask_token_id,
        max_position_embeddings=514,
        **preset,
    )


def _language_indices(language_ids: torch.Tensor, language_id: int) -> torch.Tensor:
    return language_ids.eq(language_id).nonzero(as_tuple=True)[0]


def _make_attention_config(config: DualLanguageModelConfig) -> ExperimentalBackboneConfig:
    return ExperimentalBackboneConfig(
        vocab_size=max(config.smiles_vocab_size, config.selfies_vocab_size),
        hidden_size=config.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        layer_norm_eps=config.layer_norm_eps,
    )


class DualLanguageTranslationDecoder(nn.Module):
    requires_target_language_ids = True

    def __init__(
        self,
        *,
        config: DualLanguageModelConfig,
        max_length: int,
        num_shared_layers: int = 2,
        local_num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_length = max_length
        self.smiles_embedding = nn.Embedding(config.smiles_vocab_size, config.hidden_size)
        self.selfies_embedding = nn.Embedding(config.selfies_vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(max_length, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(dropout)
        self.shared_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_shared_layers)
            ]
        )
        self.smiles_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(local_num_layers)
            ]
        )
        self.selfies_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(local_num_layers)
            ]
        )
        self.smiles_output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.selfies_output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.smiles_output_bias = nn.Parameter(torch.zeros(config.smiles_vocab_size))
        self.selfies_output_bias = nn.Parameter(torch.zeros(config.selfies_vocab_size))

    def _embed_targets(self, target_ids: torch.Tensor, language_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            target_ids.size(0),
            target_ids.size(1),
            self.config.hidden_size,
            device=target_ids.device,
            dtype=self.smiles_embedding.weight.dtype,
        )
        for language_id, embedding in (
            (PSMILES_LANGUAGE_ID, self.smiles_embedding),
            (PSELFIES_LANGUAGE_ID, self.selfies_embedding),
        ):
            indices = _language_indices(language_ids, language_id)
            if indices.numel() == 0:
                continue
            hidden.index_copy_(0, indices, embedding(target_ids.index_select(0, indices)))
        positions = torch.arange(target_ids.size(1), device=target_ids.device).unsqueeze(0)
        hidden = hidden + self.position_embedding(positions)
        return self.embedding_dropout(self.embedding_norm(hidden))

    def _padding_mask(self, token_ids: torch.Tensor, language_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for language_id, pad_id in (
            (PSMILES_LANGUAGE_ID, self.config.smiles_pad_id),
            (PSELFIES_LANGUAGE_ID, self.config.selfies_pad_id),
        ):
            row_mask = language_ids.eq(language_id)
            if row_mask.any():
                padding_mask[row_mask] = token_ids[row_mask].eq(pad_id)
        return padding_mask

    def _apply_language_decoder_layers(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        target_padding_mask: torch.Tensor,
        language_ids: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = hidden_states.clone()
        for language_id, layers in (
            (PSMILES_LANGUAGE_ID, self.smiles_layers),
            (PSELFIES_LANGUAGE_ID, self.selfies_layers),
        ):
            indices = _language_indices(language_ids, language_id)
            if indices.numel() == 0:
                continue
            lang_hidden = hidden_states.index_select(0, indices)
            lang_memory = memory.index_select(0, indices)
            lang_memory_padding = memory_padding_mask.index_select(0, indices)
            lang_target_padding = target_padding_mask.index_select(0, indices)
            for layer in layers:
                lang_hidden = layer(
                    tgt=lang_hidden,
                    memory=lang_memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=lang_target_padding,
                    memory_key_padding_mask=lang_memory_padding,
                )
            output.index_copy_(0, indices, lang_hidden)
        return output

    def _project(self, hidden_states: torch.Tensor, language_ids: torch.Tensor) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for language_id, norm, embedding, bias in (
            (PSMILES_LANGUAGE_ID, self.smiles_output_norm, self.smiles_embedding, self.smiles_output_bias),
            (PSELFIES_LANGUAGE_ID, self.selfies_output_norm, self.selfies_embedding, self.selfies_output_bias),
        ):
            indices = _language_indices(language_ids, language_id)
            if indices.numel() == 0:
                continue
            lang_hidden = norm(hidden_states.index_select(0, indices))
            outputs[language_id] = (indices, F.linear(lang_hidden, embedding.weight, bias))
        return outputs

    def forward(
        self,
        memory: torch.Tensor,
        memory_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        target_language_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:].clone()
        for language_id, pad_id in (
            (PSMILES_LANGUAGE_ID, self.config.smiles_pad_id),
            (PSELFIES_LANGUAGE_ID, self.config.selfies_pad_id),
        ):
            row_mask = target_language_ids.eq(language_id)
            if row_mask.any():
                labels[row_mask] = labels[row_mask].masked_fill(labels[row_mask].eq(pad_id), -100)

        hidden_states = self._embed_targets(decoder_input_ids, target_language_ids)
        target_padding_mask = self._padding_mask(decoder_input_ids, target_language_ids)
        memory_padding_mask = ~memory_attention_mask.bool()
        seq_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=decoder_input_ids.device, dtype=torch.bool), diagonal=1)
        for layer in self.shared_layers:
            hidden_states = layer(
                tgt=hidden_states,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=memory_padding_mask,
            )
        hidden_states = self._apply_language_decoder_layers(
            hidden_states,
            memory,
            memory_padding_mask,
            target_padding_mask,
            target_language_ids,
            causal_mask,
        )

        projected = self._project(hidden_states, target_language_ids)
        total_loss = hidden_states.new_tensor(0.0)
        total_tokens = hidden_states.new_tensor(0.0)
        total_correct = hidden_states.new_tensor(0.0)
        for language_id, (indices, logits) in projected.items():
            lang_labels = labels.index_select(0, indices)
            valid = lang_labels.ne(-100)
            if not valid.any():
                continue
            total_loss = total_loss + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                lang_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            predictions = logits.argmax(dim=-1)
            total_correct = total_correct + (predictions.eq(lang_labels) & valid).sum()
            total_tokens = total_tokens + valid.sum()
        loss = total_loss / total_tokens.clamp(min=1.0)
        token_accuracy = total_correct / total_tokens.clamp(min=1.0)
        return loss, None, token_accuracy


class DualLanguageSharedBackbone(nn.Module):
    requires_language_ids = True

    def __init__(self, config: DualLanguageModelConfig) -> None:
        super().__init__()
        self.model_config = config
        self.config = SimpleNamespace(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
        )
        attention_config = _make_attention_config(config)
        shared_layers = max(config.num_hidden_layers - config.local_encoder_layers, 1)
        self.smiles_embeddings = nn.Embedding(config.smiles_vocab_size, config.hidden_size)
        self.selfies_embeddings = nn.Embedding(config.selfies_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.smiles_encoder_layers = nn.ModuleList([ExperimentalEncoderLayer(attention_config) for _ in range(config.local_encoder_layers)])
        self.selfies_encoder_layers = nn.ModuleList([ExperimentalEncoderLayer(attention_config) for _ in range(config.local_encoder_layers)])
        self.shared_encoder_layers = nn.ModuleList([ExperimentalEncoderLayer(attention_config) for _ in range(shared_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.smiles_mlm_layers = nn.ModuleList([ExperimentalEncoderLayer(attention_config) for _ in range(config.local_mlm_decoder_layers)])
        self.selfies_mlm_layers = nn.ModuleList([ExperimentalEncoderLayer(attention_config) for _ in range(config.local_mlm_decoder_layers)])
        self.smiles_lm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.selfies_lm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.smiles_lm_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.selfies_lm_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.smiles_lm_bias = nn.Parameter(torch.zeros(config.smiles_vocab_size))
        self.selfies_lm_bias = nn.Parameter(torch.zeros(config.selfies_vocab_size))

    def _embed_inputs(self, input_ids: torch.Tensor, language_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            self.model_config.hidden_size,
            device=input_ids.device,
            dtype=self.smiles_embeddings.weight.dtype,
        )
        for language_id, embedding in (
            (PSMILES_LANGUAGE_ID, self.smiles_embeddings),
            (PSELFIES_LANGUAGE_ID, self.selfies_embeddings),
        ):
            indices = _language_indices(language_ids, language_id)
            if indices.numel() == 0:
                continue
            hidden.index_copy_(0, indices, embedding(input_ids.index_select(0, indices)))
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        hidden = hidden + self.position_embeddings(positions)
        return self.embedding_dropout(self.embedding_norm(hidden))

    def _apply_language_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor,
        smiles_blocks: nn.ModuleList,
        selfies_blocks: nn.ModuleList,
    ) -> torch.Tensor:
        output = hidden_states.clone()
        for language_id, blocks in (
            (PSMILES_LANGUAGE_ID, smiles_blocks),
            (PSELFIES_LANGUAGE_ID, selfies_blocks),
        ):
            indices = _language_indices(language_ids, language_id)
            if indices.numel() == 0:
                continue
            lang_hidden = hidden_states.index_select(0, indices)
            lang_attention_mask = attention_mask.index_select(0, indices)
            for block in blocks:
                lang_hidden = block(lang_hidden, lang_attention_mask)
            output.index_copy_(0, indices, lang_hidden)
        return output

    def encode_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if language_ids is None:
            language_ids = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.long)
        hidden_states = self._embed_inputs(input_ids, language_ids)
        hidden_states = self._apply_language_blocks(
            hidden_states,
            attention_mask,
            language_ids,
            self.smiles_encoder_layers,
            self.selfies_encoder_layers,
        )
        for layer in self.shared_encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.final_norm(hidden_states)

    def _mlm_logits(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        adapted = self._apply_language_blocks(
            hidden_states,
            attention_mask,
            language_ids,
            self.smiles_mlm_layers,
            self.selfies_mlm_layers,
        )
        outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for language_id, dense, norm, embedding, bias in (
            (PSMILES_LANGUAGE_ID, self.smiles_lm_dense, self.smiles_lm_norm, self.smiles_embeddings, self.smiles_lm_bias),
            (PSELFIES_LANGUAGE_ID, self.selfies_lm_dense, self.selfies_lm_norm, self.selfies_embeddings, self.selfies_lm_bias),
        ):
            indices = _language_indices(language_ids, language_id)
            if indices.numel() == 0:
                continue
            lang_hidden = adapted.index_select(0, indices)
            lang_hidden = norm(F.gelu(dense(lang_hidden)))
            outputs[language_id] = (indices, F.linear(lang_hidden, embedding.weight, bias))
        return outputs

    def build_translation_decoder(
        self,
        *,
        translation_max_length: int,
        translation_decoder_layers: int,
        translation_decoder_dropout: float,
        translation_decoder_type: str,
    ) -> DualLanguageTranslationDecoder:
        if translation_decoder_type != "autoregressive":
            raise ValueError("dual_deepchem_pselfies_shared only supports autoregressive translation decoding")
        return DualLanguageTranslationDecoder(
            config=self.model_config,
            max_length=translation_max_length,
            num_shared_layers=translation_decoder_layers,
            local_num_layers=self.model_config.local_translation_decoder_layers,
            dropout=translation_decoder_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
        language_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> SimpleNamespace:
        if language_ids is None:
            language_ids = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.long)
        hidden_states = self.encode_hidden(input_ids=input_ids, attention_mask=attention_mask, language_ids=language_ids)
        loss = None
        if labels is not None:
            logits_by_language = self._mlm_logits(hidden_states, attention_mask, language_ids)
            total_loss = hidden_states.new_tensor(0.0)
            total_tokens = hidden_states.new_tensor(0.0)
            for _, (indices, logits) in logits_by_language.items():
                lang_labels = labels.index_select(0, indices)
                valid = lang_labels.ne(-100)
                if not valid.any():
                    continue
                total_loss = total_loss + F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    lang_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_tokens = total_tokens + valid.sum()
            loss = total_loss / total_tokens.clamp(min=1.0)
        output = SimpleNamespace(loss=loss, logits=None, last_hidden_state=hidden_states)
        return output if return_dict else output


def build_dual_language_backbone(
    scratch_variant: str = "base",
    max_len: int = 514,
    use_original_deepchem: bool = False,
) -> DualLanguageSharedBackbone:
    bundle = DualTokenizerBundle.load(max_len=max_len, use_original_deepchem=use_original_deepchem)
    config = build_dual_language_config(bundle, scratch_variant=scratch_variant)
    return DualLanguageSharedBackbone(config)
