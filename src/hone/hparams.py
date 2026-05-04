"""Hyperparameter loading for LoopLM / Hone.

Layered merge order:
  1. DEFAULT_HPARAMS  (lowest priority)
  2. hparams/hparams.json  (must define model_size)
  3. hparams/{model_size}.json
  4. hparams/hparams-local-run.json  (optional, highest priority)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from transformers.models.auto.tokenization_auto import AutoTokenizer

from .logging import logger
from .model import LoopLMConfig

DEFAULT_HPARAMS = {
    "spec_version": 1,
    "project": "hone",

    "sequence_length": 4096,
    "pages_per_window": 2,
    "batch_size": 8,
    "learning_rate": 0.001,

    "blocks_per_window": 2,
    "windows_per_weights": 10,

    "momentum_decay": 0.999,
    "topk_compression": 32,
    "target_chunk": 64,
    "scores_alpha": 0.001,

    "tokenizer_name": "HuggingFaceTB/SmolLM2-135M",
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "intermediate_size": 5504,
    "activation_function": "swiGLU",
    "max_position_embeddings": 4096,
    "vocab_size": 49152,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,

    "bucket_name": "your-default-bucket-name",

    "warmup_steps": 250,
    "alpha_f": 0.1,
    "t_max_scheduler": 20000,
    "outer_steps_per_shard": 455,
    "eval_lr_factor": 0.5,
}


def create_namespace(hparams: dict) -> SimpleNamespace:
    """Build a SimpleNamespace with model_config and tokenizer attached."""
    full = DEFAULT_HPARAMS.copy()
    full.update(hparams)
    ns = SimpleNamespace(**full)

    # Expose FSDP sub-dict as namespace
    if isinstance(full.get("fsdp"), dict):
        ns.fsdp = SimpleNamespace(**full["fsdp"])

    # Load tokenizer
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    ns.tokenizer = AutoTokenizer.from_pretrained(
        ns.tokenizer_name,
        verbose=False,
        clean_up_tokenization_spaces=True,
        token=token,
    )
    ns.tokenizer.pad_token = ns.tokenizer.eos_token

    ns.model_config = LoopLMConfig(
        vocab_size=getattr(ns, "vocab_size", ns.tokenizer.vocab_size),
        dim=ns.hidden_size,
        n_layers=ns.num_hidden_layers,
        n_heads=ns.num_attention_heads,
        n_kv_heads=getattr(ns, "num_key_value_heads", ns.num_attention_heads),
        intermediate_size=getattr(ns, "intermediate_size", None),
        norm_eps=getattr(ns, "rms_norm_eps", 1e-5),
        rope_theta=getattr(ns, "rope_theta", 10000.0),
        max_seq_len=ns.sequence_length,
        tie_embeddings=getattr(ns, "tie_embeddings", True),
        use_moe=getattr(ns, "use_moe", False),
        num_experts=getattr(ns, "num_experts", 8),
        moe_top_k=getattr(ns, "moe_top_k", 2),
        moe_intermediate_size=getattr(ns, "moe_intermediate_size", None),
        shared_expert_intermediate_size=getattr(ns, "shared_expert_intermediate_size", None),
        moe_layers=getattr(ns, "moe_layers", None),
        moe_aux_loss_coeff=getattr(ns, "moe_aux_loss_coeff", 0.01),
    )

    return ns


def load_hparams(
    hparams_dir: str = "hparams",
    use_local_run_hparams: bool = False,
) -> SimpleNamespace:
    """Load and merge hyperparameters from disk."""
    hparams = DEFAULT_HPARAMS.copy()
    hparams_dir_path = Path(hparams_dir)

    # 1. Base hparams
    base_file = hparams_dir_path / "hparams.json"
    try:
        with open(base_file) as f:
            hparams.update(json.load(f))
        logger.info(f"Loaded base config from {base_file}")
    except FileNotFoundError:
        logger.error(f"Base config not found: {base_file}")
        raise

    # 2. Model-size specific
    model_size = hparams.get("model_size")
    if not model_size:
        raise ValueError(f"'model_size' must be defined in {base_file}")

    model_file = hparams_dir_path / f"{model_size}.json"
    try:
        with open(model_file) as f:
            hparams.update(json.load(f))
        logger.info(f"Loaded model config from {model_file}")
    except FileNotFoundError:
        logger.error(f"Model config not found: {model_file}")
        raise ValueError(f"No hparams file for model_size '{model_size}'")

    # 3. Optional local overrides
    if use_local_run_hparams:
        local_file = hparams_dir_path / "hparams-local-run.json"
        try:
            with open(local_file) as f:
                overrides = json.load(f)
                hparams.update(overrides)
            logger.info(f"Applied local overrides from {local_file}")
        except FileNotFoundError:
            logger.warning(f"Local run file not found: {local_file}")

    logger.info(
        f"Project: '{hparams.get('project')}', "
        f"Model size: '{hparams.get('model_size')}'"
    )
    return create_namespace(hparams)


def hparams_to_json_dict(ns: SimpleNamespace) -> dict:
    """Extract a JSON-serializable dict from a hparams namespace.

    Skips non-serializable objects like tokenizer, model_config, and FSDP
    sub-namespaces (the latter is converted back to a plain dict).
    """
    out: dict = {}
    for k, v in vars(ns).items():
        if k in ("tokenizer", "model_config"):
            continue
        if isinstance(v, SimpleNamespace):
            out[k] = vars(v)
        elif isinstance(v, (str, int, float, bool, type(None), list)):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = v
    return out
