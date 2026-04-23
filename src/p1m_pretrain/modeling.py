from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .upstream import load_backbone_model


def masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> tuple[torch.Tensor, torch.Tensor]:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    top1 = (logits.argmax(dim=1) == labels).float().mean()
    return loss, top1


class TranslationDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        nhead: int,
        vocab_size: int,
        max_length: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        memory: torch.Tensor,
        memory_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:].clone()
        labels[labels == self.pad_id] = -100

        positions = torch.arange(decoder_input_ids.size(1), device=decoder_input_ids.device).unsqueeze(0)
        embedded = self.embedding(decoder_input_ids) + self.position_embedding(positions)
        seq_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_input_ids.device, dtype=torch.bool), diagonal=1
        )
        decoded = self.decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input_ids.eq(self.pad_id),
            memory_key_padding_mask=~memory_attention_mask.bool(),
        )
        logits = self.output(decoded)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        predictions = logits.argmax(dim=-1)
        valid = labels.ne(-100)
        token_accuracy = (predictions.eq(labels) & valid).sum().float() / valid.sum().clamp(min=1)
        return loss, logits, token_accuracy


class DenoisingTranslationDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        nhead: int,
        vocab_size: int,
        max_length: int,
        pad_id: int,
        mask_id: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_diffusion_steps: int = 16,
        max_corrupt_prob: float = 0.8,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.max_length = max_length
        self.num_diffusion_steps = num_diffusion_steps
        self.max_corrupt_prob = max_corrupt_prob
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        self.timestep_embedding = nn.Embedding(num_diffusion_steps + 1, hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, vocab_size)

    def _corrupt_targets(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = labels.size(0)
        device = labels.device
        timesteps = torch.randint(1, self.num_diffusion_steps + 1, (batch_size,), device=device)
        corruption = (timesteps.float() / float(self.num_diffusion_steps)) * self.max_corrupt_prob
        corruption = corruption.unsqueeze(1)
        valid = labels.ne(self.pad_id)
        noise = torch.rand_like(labels.float()) < corruption
        mask = noise & valid
        noisy = labels.clone()
        noisy[mask] = self.mask_id
        return noisy, timesteps

    def forward(
        self,
        memory: torch.Tensor,
        memory_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = target_ids[:, 1:].clone()
        noisy_ids, timesteps = self._corrupt_targets(labels)
        ce_labels = labels.clone()
        ce_labels[ce_labels == self.pad_id] = -100

        positions = torch.arange(noisy_ids.size(1), device=noisy_ids.device).unsqueeze(0)
        timestep_emb = self.timestep_embedding(timesteps).unsqueeze(1)
        embedded = self.embedding(noisy_ids) + self.position_embedding(positions) + timestep_emb
        decoded = self.decoder(
            tgt=embedded,
            memory=memory,
            tgt_key_padding_mask=noisy_ids.eq(self.pad_id),
            memory_key_padding_mask=~memory_attention_mask.bool(),
        )
        logits = self.output(decoded)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ce_labels.reshape(-1), ignore_index=-100)
        predictions = logits.argmax(dim=-1)
        valid = ce_labels.ne(-100)
        token_accuracy = (predictions.eq(ce_labels) & valid).sum().float() / valid.sum().clamp(min=1)
        return loss, logits, token_accuracy


@dataclass
class LossBundle:
    total_loss: torch.Tensor
    mlm_loss: torch.Tensor
    view_loss: torch.Tensor
    translation_loss: torch.Tensor
    view_top1: torch.Tensor
    translation_token_accuracy: torch.Tensor


class ContinuationModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        init_mode: str,
        pretrain_objective: str = "mlm",
        translation_vocab_size: int,
        translation_pad_id: int,
        translation_bos_id: int,
        translation_eos_id: int,
        translation_max_length: int,
        translation_mask_id: int,
        translation_decoder_type: str = "autoregressive",
        translation_decoder_layers: int = 2,
        translation_decoder_dropout: float = 0.1,
        translation_num_diffusion_steps: int = 16,
        translation_diffusion_max_corrupt_prob: float = 0.8,
        view_temperature: float = 0.07,
        force_generic_translation_decoder: bool = False,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.init_mode = init_mode
        if pretrain_objective not in {"mlm", "t5_span_infilling"}:
            raise ValueError(f"Unsupported pretrain_objective: {pretrain_objective}")
        self.pretrain_objective = pretrain_objective
        self.view_temperature = view_temperature
        self.backbone = load_backbone_model(backbone_name, init_mode=init_mode, **(backbone_kwargs or {}))
        hidden_size = self.backbone.config.hidden_size
        num_heads = self.backbone.config.num_attention_heads
        self.view_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        if hasattr(self.backbone, "build_translation_decoder") and not force_generic_translation_decoder:
            self.translation_decoder = self.backbone.build_translation_decoder(
                translation_max_length=translation_max_length,
                translation_decoder_layers=translation_decoder_layers,
                translation_decoder_dropout=translation_decoder_dropout,
                translation_decoder_type=translation_decoder_type,
            )
        else:
            if translation_decoder_type == "diffusion_like":
                self.translation_decoder = DenoisingTranslationDecoder(
                    hidden_size=hidden_size,
                    nhead=num_heads,
                    vocab_size=translation_vocab_size,
                    max_length=translation_max_length,
                    pad_id=translation_pad_id,
                    mask_id=translation_mask_id,
                    num_layers=translation_decoder_layers,
                    dropout=translation_decoder_dropout,
                    num_diffusion_steps=translation_num_diffusion_steps,
                    max_corrupt_prob=translation_diffusion_max_corrupt_prob,
                )
            else:
                self.translation_decoder = TranslationDecoder(
                    hidden_size=hidden_size,
                    nhead=num_heads,
                    vocab_size=translation_vocab_size,
                    max_length=translation_max_length,
                    pad_id=translation_pad_id,
                    bos_id=translation_bos_id,
                    eos_id=translation_eos_id,
                    num_layers=translation_decoder_layers,
                    dropout=translation_decoder_dropout,
                )
        self.span_infilling_decoder = None
        if self.pretrain_objective == "t5_span_infilling":
            self.span_infilling_decoder = self._build_span_infilling_decoder(
                max_length=translation_max_length,
                num_layers=translation_decoder_layers,
                dropout=translation_decoder_dropout,
                hidden_size=hidden_size,
                num_heads=num_heads,
            )

    def _build_span_infilling_decoder(
        self,
        *,
        max_length: int,
        num_layers: int,
        dropout: float,
        hidden_size: int,
        num_heads: int,
    ) -> nn.Module:
        if hasattr(self.backbone, "build_translation_decoder"):
            return self.backbone.build_translation_decoder(
                translation_max_length=max_length,
                translation_decoder_layers=num_layers,
                translation_decoder_dropout=dropout,
                translation_decoder_type="autoregressive",
            )
        model_config = getattr(self.backbone, "model_config", None)
        if model_config is None or not hasattr(model_config, "selfies_vocab_size"):
            raise ValueError(
                f"pretrain_objective='t5_span_infilling' requires a dual backbone with pSELFIES vocab metadata, got {self.backbone_name}"
            )
        return TranslationDecoder(
            hidden_size=hidden_size,
            nhead=num_heads,
            vocab_size=model_config.selfies_vocab_size,
            max_length=max_length,
            pad_id=model_config.selfies_pad_id,
            bos_id=model_config.selfies_cls_id,
            eos_id=model_config.selfies_sep_id,
            num_layers=num_layers,
            dropout=dropout,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.backbone, "encode_hidden"):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if language_ids is not None:
                kwargs["language_ids"] = language_ids
            hidden = self.backbone.encode_hidden(**kwargs)
        else:
            outputs = self.backbone.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden = outputs.last_hidden_state
        pooled = masked_mean(hidden, attention_mask)
        return hidden, pooled

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        view_weight: float | torch.Tensor | None = None,
        translation_weight: float | torch.Tensor | None = None,
    ) -> LossBundle:
        # Support weights passed in batch dict (for DataParallel compatibility)
        if view_weight is None:
            view_weight = float(batch.pop("_view_weight").flatten()[0])
        elif isinstance(view_weight, torch.Tensor):
            view_weight = float(view_weight.flatten()[0])
        if translation_weight is None:
            translation_weight = float(batch.pop("_translation_weight").flatten()[0])
        elif isinstance(translation_weight, torch.Tensor):
            translation_weight = float(translation_weight.flatten()[0])
        if self.pretrain_objective == "t5_span_infilling":
            if self.span_infilling_decoder is None:
                raise RuntimeError("span_infilling_decoder is not initialized")
            mlm_memory, _ = self.encode(
                batch["mlm_input_ids"],
                batch["mlm_attention_mask"],
                batch.get("mlm_language_ids"),
            )
            infill_kwargs = {}
            if getattr(self.span_infilling_decoder, "requires_target_language_ids", False):
                infill_kwargs["target_language_ids"] = batch["span_target_language_ids"]
            mlm_loss, _, _ = self.span_infilling_decoder(
                memory=mlm_memory,
                memory_attention_mask=batch["mlm_attention_mask"],
                target_ids=batch["span_target_ids"],
                **infill_kwargs,
            )
        else:
            mlm_kwargs = {}
            if "mlm_language_ids" in batch:
                mlm_kwargs["language_ids"] = batch["mlm_language_ids"]
            mlm_outputs = self.backbone(
                input_ids=batch["mlm_input_ids"],
                attention_mask=batch["mlm_attention_mask"],
                labels=batch["mlm_labels"],
                return_dict=True,
                **mlm_kwargs,
            )
            mlm_loss = mlm_outputs.loss

        view1_hidden, view1_pooled = self.encode(
            batch["view1_input_ids"],
            batch["view1_attention_mask"],
            batch.get("view1_language_ids"),
        )
        view2_hidden, view2_pooled = self.encode(
            batch["view2_input_ids"],
            batch["view2_attention_mask"],
            batch.get("view2_language_ids"),
        )
        del view1_hidden, view2_hidden
        view_loss, view_top1 = info_nce_loss(
            self.view_projector(view1_pooled),
            self.view_projector(view2_pooled),
            temperature=self.view_temperature,
        )

        translation_memory, _ = self.encode(
            batch["translation_input_ids"],
            batch["translation_attention_mask"],
            batch.get("translation_source_language_ids"),
        )
        decoder_kwargs = {}
        if getattr(self.translation_decoder, "requires_target_language_ids", False):
            decoder_kwargs["target_language_ids"] = batch["translation_target_language_ids"]
        translation_loss, _, translation_token_accuracy = self.translation_decoder(
            memory=translation_memory,
            memory_attention_mask=batch["translation_attention_mask"],
            target_ids=batch["translation_target_ids"],
            **decoder_kwargs,
        )

        total_loss = mlm_loss + view_weight * view_loss + translation_weight * translation_loss
        # Return stacked tensor for DataParallel gather compatibility
        metrics = torch.stack([
            total_loss,
            mlm_loss.detach(),
            view_loss.detach(),
            translation_loss.detach(),
            view_top1.detach(),
            translation_token_accuracy.detach(),
        ])
        return metrics
