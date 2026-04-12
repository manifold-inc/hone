"""HF causal LM factory with optional FSDP2 sharding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

if TYPE_CHECKING:
    from hone.config import HoneConfig

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, tuple[str, str] | None] = {
    "llama": ("LlamaForCausalLM", "LlamaConfig"),
    "mistral": ("MistralForCausalLM", "MistralConfig"),
    "qwen2": ("Qwen2ForCausalLM", "Qwen2Config"),
    "gemma": ("GemmaForCausalLM", "GemmaConfig"),
    "ouro": None,
}


def build_model(config: HoneConfig) -> nn.Module:
    """Instantiate a causal LM from the registry or HF auto classes."""
    model_type = config.model_type.lower()

    if model_type == "ouro":
        if not config.model_name_or_path:
            raise ValueError("model_name_or_path is required when model_type is 'ouro'")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )
        logger.info("Loaded ouro-style model from %s", config.model_name_or_path)
        return _maybe_fsdp(model, config)

    entry = MODEL_REGISTRY.get(model_type)
    if entry is None:
        hf_config = AutoConfig.for_model(model_type, **config.model_args)
        model = AutoModelForCausalLM.from_config(hf_config)
        logger.info("Built AutoModelForCausalLM for unknown type %r", model_type)
        return _maybe_fsdp(model, config)

    model_cls_name, config_cls_name = entry
    model_cls = getattr(transformers, model_cls_name)
    config_cls = getattr(transformers, config_cls_name)

    if config.model_name_or_path:
        model = model_cls.from_pretrained(config.model_name_or_path)
        logger.info("Loaded %s from %s", model_cls_name, config.model_name_or_path)
    else:
        hf_config = config_cls(**config.model_args)
        model = model_cls(hf_config)
        logger.info("Initialized %s from config args", model_cls_name)

    return _maybe_fsdp(model, config)


def _maybe_fsdp(model: nn.Module, config: HoneConfig) -> nn.Module:
    if config.fsdp_enabled:
        return apply_fsdp(model, config)
    return model


def apply_fsdp(model: nn.Module, config: HoneConfig) -> nn.Module:
    """Shard decoder blocks then the root module using FSDP2 ``fully_shard``."""
    if not config.fsdp_enabled or not torch.distributed.is_initialized():
        return model

    from torch.distributed.fsdp import fully_shard

    layers = get_decoder_layers(model)
    if not layers:
        logger.warning("No decoder layers found; applying fully_shard to root only")
        fully_shard(model)
        return model

    for layer in layers:
        fully_shard(layer)
    fully_shard(model)
    return model


def get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return stacked transformer decoder blocks for common HF layouts."""
    candidates = (
        "layers",
        "h",
        "blocks",
    )
    roots = (model, getattr(model, "model", None), getattr(model, "transformer", None))

    for root in roots:
        if root is None:
            continue
        inner = getattr(root, "model", root)
        for name in candidates:
            seq = getattr(inner, name, None)
            if isinstance(seq, nn.ModuleList):
                return list(seq)

        gpt_neox = getattr(inner, "gpt_neox", None)
        if gpt_neox is not None:
            seq = getattr(gpt_neox, "layers", None)
            if isinstance(seq, nn.ModuleList):
                return list(seq)

    return []
