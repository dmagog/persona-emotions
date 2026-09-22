"""One place to load a model and read its config.

This replaces eleven `from_pretrained` calls with differing arguments. That
spread once produced pairs generated in bf16 while activations were captured in
fp16, so a direction was extracted in a different numeric regime than the one it
was later applied in.

It also decides the dtype. Forcing bf16 does not work: Turing cards (RTX 2070,
sm75) have no hardware support for it. Forcing fp16 does not work either:
models trained in bf16, such as Gemma and part of the Llama family, overflow in
fp16, whose range ends at 65504, while bf16 covers the same magnitudes as fp32.
So `auto` looks at both the hardware and the training dtype, and states out loud
what it picked and why.

Usage:
    from emotion.loader import LoadSpec, load_model_and_tokenizer, resolve_layers
    model, tok, info = load_model_and_tokenizer(LoadSpec(hf_id="Qwen/Qwen3-1.7B"))
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class LoadSpec:
    """Everything that affects loading. The fields mirror the `load` config section."""

    hf_id: str
    dtype: str = "auto"
    device_map: str | dict | None = "auto"
    max_memory: dict | None = None
    offload_folder: str | None = None
    trust_remote_code: bool = False
    load_in_4bit: bool = False
    seed: int = 0
    attn_implementation: str | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "LoadSpec":
        load = dict(cfg.get("load") or {})
        # yaml gives max_memory string keys, torch wants int device numbers
        mm = load.get("max_memory")
        if isinstance(mm, dict):
            load["max_memory"] = {
                (int(k) if str(k).isdigit() else k): v for k, v in mm.items()
            }
        return cls(hf_id=cfg["hf_id"], **load)


def set_determinism(seed: int = 0) -> None:
    """Seed every source of randomness. Determinism used to rest on greedy decoding alone."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def bf16_supported() -> bool:
    """Report whether bf16 exists in HARDWARE. On Turing (sm75, RTX 2070) it does not.

    `torch.cuda.is_bf16_supported()` also answers True where bf16 is merely
    emulated in software. On the 2070 (sm 7.5) it returned True, and two runs of
    the grid went to emulated bf16: slower than the rest, and with different low
    bits in the activations. This asks the hardware directly and treats the
    torch answer only as an upper bound.
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    if major < 8:
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return True


def trained_dtype(config) -> torch.dtype | None:
    """Report the dtype a model was trained in. transformers stores it under varying keys."""
    for attr in ("torch_dtype", "dtype"):
        v = getattr(config, attr, None)
        if isinstance(v, torch.dtype):
            return v
        if isinstance(v, str) and v in DTYPES:
            return DTYPES[v]
    return None


def resolve_dtype(spec_dtype: str, config) -> tuple[torch.dtype, str]:
    """Pick a dtype and explain the choice. Returns (dtype, reason)."""
    if spec_dtype and spec_dtype != "auto":
        key = spec_dtype.lower()
        if key not in DTYPES:
            raise ValueError(f"unknown dtype {spec_dtype!r}, expected one of {sorted(DTYPES)}")
        return DTYPES[key], f"set in the config ({spec_dtype})"

    trained = trained_dtype(config)
    if trained is torch.bfloat16:
        if bf16_supported():
            return torch.bfloat16, "trained in bf16 and the card supports it"
        return torch.float16, (
            "trained in bf16, but the card cannot do it (Turing), so fp16 was used. "
            "WARNING: this may overflow; check the vectors for finite values"
        )
    if trained is torch.float32:
        return torch.float16, "trained in fp32, fp16 used for inference"
    return torch.float16, "training dtype unknown, fp16 used"


def text_config(config):
    """Return the language-side config. Multimodal models nest it."""
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            sub = getter()
            if sub is not None:
                return sub
        except Exception:
            pass
    for attr in ("text_config", "llm_config", "language_config"):
        sub = getattr(config, attr, None)
        if sub is not None:
            return sub
    return config


def num_layers(config) -> int:
    cfg = config
    for candidate in (config, text_config(config)):
        n = getattr(candidate, "num_hidden_layers", None)
        if isinstance(n, int):
            return n
        cfg = candidate
    raise ValueError(
        f"num_hidden_layers is absent from both the top level and the nested config "
        f"({type(config).__name__}). This is probably an unusual architecture, so the "
        "model config needs an explicit layer list."
    )


def hidden_size(config) -> int:
    for candidate in (config, text_config(config)):
        h = getattr(candidate, "hidden_size", None)
        if isinstance(h, int):
            return h
    raise ValueError(f"hidden_size is absent from {type(config).__name__}")


def resolve_layers(config, layers) -> list[int]:
    """Resolve candidate layers from depth fractions (0 < x < 1) or absolute indices.

    Fractions carry across models: 0.45 of 28 layers gives 13, of 36 gives 16.
    Absolute indices tie a config to one specific model.
    """
    n = num_layers(config)
    out: list[int] = []
    for x in layers:
        idx = max(1, round(n * float(x))) if 0 < float(x) < 1 else int(x)
        if not 0 <= idx < n:
            raise ValueError(f"layer {idx} is outside 0..{n - 1} (the model has {n} layers)")
        out.append(idx)
    return sorted(set(out))


def load_model_and_tokenizer(spec: LoadSpec, for_generation: bool = True):
    """Load a model and tokenizer. Returns (model, tokenizer, info).

    info goes into the run manifest, so the numeric regime stays recoverable.
    """
    set_determinism(spec.seed)
    config = AutoConfig.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code, token=os.environ.get("HF_TOKEN")
    )
    dtype, why = resolve_dtype(spec.dtype, config)

    kwargs = dict(
        torch_dtype=dtype,
        device_map=spec.device_map,
        trust_remote_code=spec.trust_remote_code,
        token=os.environ.get("HF_TOKEN"),
    )
    if spec.max_memory:
        kwargs["max_memory"] = spec.max_memory
    if spec.offload_folder:
        kwargs["offload_folder"] = spec.offload_folder
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation
    if spec.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )

    print(f"[loader] {spec.hf_id}: dtype={dtype}, {why}", flush=True)
    if spec.load_in_4bit:
        print("[loader] WARNING: 4-bit. Activations differ from full precision, so "
              "vectors from a quantized model are not comparable with the rest.", flush=True)

    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    model.eval()

    tok = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code, token=os.environ.get("HF_TOKEN")
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    if for_generation:
        tok.padding_side = "left"  # decoder-only: right padding shifts the answers

    devices = {str(p.device) for p in model.parameters()}
    on_cpu = sum(1 for p in model.parameters() if p.device.type == "cpu")
    if on_cpu and torch.cuda.is_available():
        print(f"[loader] WARNING: {on_cpu} tensors on the CPU, the card ran out of memory. "
              "Set max_memory or offload_folder in the config.", flush=True)

    info = {
        "hf_id": spec.hf_id,
        "dtype": str(dtype),
        "dtype_reason": why,
        "devices": sorted(devices),
        "num_layers": num_layers(config),
        "hidden_size": hidden_size(config),
        "seed": spec.seed,
        "load_in_4bit": spec.load_in_4bit,
        "bf16_supported": bf16_supported(),
    }
    return model, tok, info


def spec_to_dict(spec: LoadSpec) -> dict:
    return asdict(spec)
