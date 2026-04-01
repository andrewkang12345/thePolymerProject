from __future__ import annotations

import importlib.util
import json
from types import ModuleType

import torch
from transformers import RobertaConfig, RobertaForMaskedLM

from .experimental_backbone import ExperimentalBackboneConfig, ExperimentalEncoderForMLM
from .paths import get_paths


PATHS = get_paths()
TRANSPOLYMER_REPO = PATHS.transpolymer_repo
MMPOLYMER_REPO = PATHS.mmpolymer_repo

TRANSPOLYMER_CHECKPOINT = (
    PATHS.checkpoints_dir / "transpolymer" / "pytorch_model.bin"
)
MMPOLYMER_CHECKPOINT = PATHS.checkpoints_dir / "mmpolymer" / "pretrain.pt"


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_polymer_smiles_tokenizer(max_len: int = 256):
    module = _load_module(TRANSPOLYMER_REPO / "PolymerSmilesTokenization.py", "transpolymer_tokenizer")
    if not getattr(module.PolymerSmilesTokenizer, "_codex_patched", False):
        original_init = module.PolymerSmilesTokenizer.__init__

        def patched_init(self, vocab_file, merges_file, *args, **kwargs):
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                self.encoder = json.load(vocab_handle)
            self.decoder = {value: key for key, value in self.encoder.items()}
            return original_init(self, vocab_file, merges_file, *args, **kwargs)

        module.PolymerSmilesTokenizer.__init__ = patched_init
        module.PolymerSmilesTokenizer._codex_patched = True
    tokenizer = module.PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=max_len)
    return tokenizer


def _apply_scratch_variant(config: RobertaConfig, scratch_variant: str) -> RobertaConfig:
    presets = {
        "base": dict(num_hidden_layers=6, hidden_size=768, num_attention_heads=12, intermediate_size=3072),
        "deep": dict(num_hidden_layers=8, hidden_size=768, num_attention_heads=12, intermediate_size=3072),
        "small": dict(num_hidden_layers=4, hidden_size=512, num_attention_heads=8, intermediate_size=2048),
        "tiny": dict(num_hidden_layers=3, hidden_size=384, num_attention_heads=6, intermediate_size=1536),
    }
    if scratch_variant not in presets:
        raise ValueError(f"Unsupported scratch_variant: {scratch_variant}")
    preset = presets[scratch_variant]
    config.num_hidden_layers = preset["num_hidden_layers"]
    config.hidden_size = preset["hidden_size"]
    config.num_attention_heads = preset["num_attention_heads"]
    config.intermediate_size = preset["intermediate_size"]
    return config


def _build_transpolymer_scratch(scratch_variant: str = "base") -> RobertaForMaskedLM:
    config = RobertaConfig.from_pretrained(str(TRANSPOLYMER_REPO / "ckpt" / "pretrain.pt"))
    config = _apply_scratch_variant(config, scratch_variant)
    model = RobertaForMaskedLM(config)
    model.tie_weights()
    return model


def _build_mmpolymer_scratch(scratch_variant: str = "base") -> RobertaForMaskedLM:
    config = RobertaConfig.from_pretrained(str(MMPOLYMER_REPO / "MMPolymer" / "models" / "config"))
    config = _apply_scratch_variant(config, scratch_variant)
    model = RobertaForMaskedLM(config)
    model.tie_weights()
    return model


def _build_experimental_scratch(
    base_config: RobertaConfig,
    *,
    scratch_variant: str,
    position_embedding_type: str,
    attention_variant: str,
    num_key_value_heads: int,
) -> ExperimentalEncoderForMLM:
    base_config = _apply_scratch_variant(base_config, scratch_variant)
    config = ExperimentalBackboneConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        intermediate_size=base_config.intermediate_size,
        max_position_embeddings=base_config.max_position_embeddings,
        hidden_dropout_prob=base_config.hidden_dropout_prob,
        attention_probs_dropout_prob=base_config.attention_probs_dropout_prob,
        layer_norm_eps=base_config.layer_norm_eps,
        position_embedding_type=position_embedding_type,
        attention_variant=attention_variant,
        num_key_value_heads=num_key_value_heads,
    )
    return ExperimentalEncoderForMLM(config)


def _load_transpolymer() -> RobertaForMaskedLM:
    config = RobertaConfig.from_pretrained(str(TRANSPOLYMER_REPO / "ckpt" / "pretrain.pt"))
    model = RobertaForMaskedLM(config)
    state_dict = torch.load(TRANSPOLYMER_CHECKPOINT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.decoder.weight", "lm_head.decoder.bias"}
    allowed_unexpected = {"roberta.embeddings.position_ids"}
    if set(missing) - allowed_missing or set(unexpected) - allowed_unexpected:
        raise RuntimeError(f"Unexpected TransPolymer load result. missing={missing} unexpected={unexpected}")
    model.tie_weights()
    return model


def _load_mmpolymer() -> RobertaForMaskedLM:
    config = RobertaConfig.from_pretrained(str(MMPOLYMER_REPO / "MMPolymer" / "models" / "config"))
    model = RobertaForMaskedLM(config)
    raw_state = torch.load(MMPOLYMER_CHECKPOINT, map_location="cpu")["model"]
    state_dict = {}
    for key, value in raw_state.items():
        if key.startswith("PretrainedModel."):
            renamed = "roberta." + key[len("PretrainedModel.") :]
            state_dict[renamed] = value
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "lm_head.bias",
        "lm_head.dense.weight",
        "lm_head.dense.bias",
        "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias",
        "lm_head.decoder.weight",
        "lm_head.decoder.bias",
    }
    allowed_unexpected = {
        "roberta.pooler.dense.weight",
        "roberta.pooler.dense.bias",
        "roberta.embeddings.position_ids",
    }
    if set(unexpected) - allowed_unexpected:
        raise RuntimeError(f"Unexpected MMPolymer keys: {unexpected}")
    unknown_missing = set(missing) - allowed_missing
    if unknown_missing:
        raise RuntimeError(f"Unexpected MMPolymer missing keys: {sorted(unknown_missing)}")
    model.tie_weights()
    return model


MOLFORMER_HF = "ibm-research/MoLFormer-XL-both-10pct"


class MoLFormerForMLM(torch.nn.Module):
    """Wrapper around MoLFormer to match the RobertaForMaskedLM interface."""

    def __init__(self, hf_model, hf_config):
        super().__init__()
        self.molformer = hf_model
        self.config = hf_config

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True, **kwargs):
        return self.molformer(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=return_dict)

    def encode_hidden(self, input_ids, attention_mask=None):
        outputs = self.molformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") and outputs.hidden_states else outputs.logits


def _patch_molformer_imports():
    """Patch transformers compatibility issues for MoLFormer's custom code."""
    import sys
    # Fix: transformers.onnx removed in newer versions
    if "transformers.onnx" not in sys.modules:
        sys.modules["transformers.onnx"] = type(sys)("fake_onnx")
        sys.modules["transformers.onnx"].OnnxConfig = object
    # Fix: find_pruneable_heads_and_indices renamed to find_prunable_heads_and_indices
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        if hasattr(pu, "find_prunable_heads_and_indices"):
            pu.find_pruneable_heads_and_indices = pu.find_prunable_heads_and_indices


def _load_molformer():
    _patch_molformer_imports()
    # Directly patch the source file's import before loading
    import importlib, types
    from huggingface_hub import hf_hub_download
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        pu.find_pruneable_heads_and_indices = getattr(pu, "find_prunable_heads_and_indices", lambda *a, **k: None)
    # Force reimport of the cached module
    import sys
    mod_keys = [k for k in sys.modules if "MoLFormer" in k or "molformer" in k.lower()]
    for k in mod_keys:
        del sys.modules[k]
    from transformers import AutoModelForMaskedLM, AutoConfig
    config = AutoConfig.from_pretrained(MOLFORMER_HF, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MOLFORMER_HF, trust_remote_code=True)
    return MoLFormerForMLM(model, config)


SMI_TED_VOCAB = PATHS.smi_ted_dir / "bert_vocab_curated.txt"


def _build_dual_deepchem_pselfies_tokenizer(max_len: int = 514):
    from .dual_tokenizer import load_deepchem_smiles_tokenizer

    return load_deepchem_smiles_tokenizer(max_len=max_len)


def _build_dual_correctdeepchem_pselfies_tokenizer(max_len: int = 514):
    from .dual_tokenizer import load_original_deepchem_smiles_tokenizer

    return load_original_deepchem_smiles_tokenizer(max_len=max_len)


def _build_dual_deepchem_pselfies_scratch(scratch_variant: str = "base"):
    from .dual_language_model import build_dual_language_backbone

    return build_dual_language_backbone(scratch_variant=scratch_variant)


def _build_dual_correctdeepchem_pselfies_scratch(scratch_variant: str = "base"):
    from .dual_language_model import build_dual_language_backbone

    return build_dual_language_backbone(scratch_variant=scratch_variant, use_original_deepchem=True)


def load_tokenizer_for_backbone(name: str, max_len: int = 256):
    """Load the appropriate tokenizer for a given backbone."""
    if name == "smi_ted":
        from .smi_ted_extended import build_extended_smi_ted_tokenizer
        return build_extended_smi_ted_tokenizer(max_len=202)
    if name == "dual_deepchem_pselfies_shared":
        return _build_dual_deepchem_pselfies_tokenizer(max_len=max_len)
    if name == "dual_correctdeepchem_pselfies_shared":
        return _build_dual_correctdeepchem_pselfies_tokenizer(max_len=max_len)
    if name in ("transpolymer", "mmpolymer"):
        return load_polymer_smiles_tokenizer(max_len=max_len)
    elif name == "molformer":
        import sys
        if "transformers.onnx" not in sys.modules:
            sys.modules["transformers.onnx"] = type(sys)("fake_onnx")
            sys.modules["transformers.onnx"].OnnxConfig = object
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MOLFORMER_HF, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown backbone for tokenizer: {name}")


def load_backbone_model(
    name: str,
    init_mode: str = "scratch",
    backbone_family: str = "upstream_roberta",
    scratch_variant: str = "base",
    position_embedding_type: str = "absolute",
    attention_variant: str = "mha",
    num_key_value_heads: int = 4,
):
    if name == "smi_ted":
        if init_mode == "scratch":
            from .smi_ted_extended import load_smi_ted_scratch_extended
            model, _ = load_smi_ted_scratch_extended()
        else:
            from .smi_ted_extended import load_smi_ted_extended
            model, _ = load_smi_ted_extended()
        return model
    if name == "molformer":
        return _load_molformer()
    if name == "dual_deepchem_pselfies_shared":
        return _build_dual_deepchem_pselfies_scratch(scratch_variant=scratch_variant)
    if name == "dual_correctdeepchem_pselfies_shared":
        return _build_dual_correctdeepchem_pselfies_scratch(scratch_variant=scratch_variant)
    if init_mode == "scratch":
        if backbone_family == "experimental":
            if name == "transpolymer":
                base_config = RobertaConfig.from_pretrained(str(TRANSPOLYMER_REPO / "ckpt" / "pretrain.pt"))
            elif name == "mmpolymer":
                base_config = RobertaConfig.from_pretrained(str(MMPOLYMER_REPO / "MMPolymer" / "models" / "config"))
            else:
                raise ValueError(f"Unsupported backbone: {name}")
            return _build_experimental_scratch(
                base_config,
                scratch_variant=scratch_variant,
                position_embedding_type=position_embedding_type,
                attention_variant=attention_variant,
                num_key_value_heads=num_key_value_heads,
            )
        if name == "transpolymer":
            return _build_transpolymer_scratch(scratch_variant=scratch_variant)
        if name == "mmpolymer":
            return _build_mmpolymer_scratch(scratch_variant=scratch_variant)
    if init_mode == "checkpoint":
        if backbone_family != "upstream_roberta":
            raise ValueError("Checkpoint mode only supports backbone_family='upstream_roberta'")
        if name == "transpolymer":
            return _load_transpolymer()
        if name == "mmpolymer":
            return _load_mmpolymer()
    raise ValueError(f"Unsupported backbone: {name}")
