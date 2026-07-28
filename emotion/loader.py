"""Единая загрузка модели и разбор её конфига.

Одно место вместо одиннадцати вызовов `from_pretrained` с разными параметрами.
Закрывает разнобой, из-за которого пары генерировались в bf16, а активации
снимались в fp16 — то есть вектор снимался в другом численном режиме, чем тот,
в котором его потом применяли.

Отдельно решает выбор dtype. Просто поставить bf16 нельзя: на Turing (RTX 2070,
sm75) аппаратной поддержки bf16 нет. Просто поставить fp16 тоже нельзя: модели,
обученные в bf16 (gemma, часть Llama), в fp16 переполняются — диапазон fp16
кончается на 65504, а bf16 держит те же порядки, что fp32. Поэтому режим `auto`
смотрит и на железо, и на dtype обучения, и громко пишет, что выбрал и почему.

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
    """Всё, что влияет на загрузку. Поля соответствуют секции `load` конфига модели."""

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
        # max_memory приходит из yaml с ключами-строками, torch ждёт int для номеров карт
        mm = load.get("max_memory")
        if isinstance(mm, dict):
            load["max_memory"] = {
                (int(k) if str(k).isdigit() else k): v for k, v in mm.items()
            }
        return cls(hf_id=cfg["hf_id"], **load)


def set_determinism(seed: int = 0) -> None:
    """Сиды. В проекте их не задавали нигде — детерминизм держался только на greedy."""
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
    """Есть ли аппаратный bf16. На Turing (sm75, RTX 2070) — нет."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8


def trained_dtype(config) -> torch.dtype | None:
    """В каком типе модель обучалась. transformers кладёт это в конфиг по-разному."""
    for attr in ("torch_dtype", "dtype"):
        v = getattr(config, attr, None)
        if isinstance(v, torch.dtype):
            return v
        if isinstance(v, str) and v in DTYPES:
            return DTYPES[v]
    return None


def resolve_dtype(spec_dtype: str, config) -> tuple[torch.dtype, str]:
    """Выбрать тип и объяснить выбор. Возвращает (dtype, причина)."""
    if spec_dtype and spec_dtype != "auto":
        key = spec_dtype.lower()
        if key not in DTYPES:
            raise ValueError(f"неизвестный dtype {spec_dtype!r}, допустимы {sorted(DTYPES)}")
        return DTYPES[key], f"задан в конфиге ({spec_dtype})"

    trained = trained_dtype(config)
    if trained is torch.bfloat16:
        if bf16_supported():
            return torch.bfloat16, "модель обучена в bf16, карта его поддерживает"
        return torch.float16, (
            "модель обучена в bf16, но карта его не умеет (Turing) — взят fp16. "
            "ВНИМАНИЕ: возможно переполнение, проверяйте векторы на конечность"
        )
    if trained is torch.float32:
        return torch.float16, "модель в fp32, для инференса взят fp16"
    return torch.float16, "dtype обучения неизвестен, взят fp16"


def text_config(config):
    """Конфиг языковой части. У мультимодальных всё лежит во вложенном."""
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
        f"num_hidden_layers не найден ни на верхнем уровне, ни во вложенном конфиге "
        f"({type(config).__name__}). Вероятно нестандартная архитектура — "
        "нужен явный список слоёв в конфиге модели."
    )


def hidden_size(config) -> int:
    for candidate in (config, text_config(config)):
        h = getattr(candidate, "hidden_size", None)
        if isinstance(h, int):
            return h
    raise ValueError(f"hidden_size не найден в {type(config).__name__}")


def resolve_layers(config, layers) -> list[int]:
    """Кандидаты слоёв: доли глубины (0 < x < 1) или абсолютные номера.

    Доли переносимы между моделями: 0.45 от 28 слоёв даёт 13, от 36 — 16.
    Абсолютные номера привязывают конфиг к конкретной модели.
    """
    n = num_layers(config)
    out: list[int] = []
    for x in layers:
        idx = max(1, round(n * float(x))) if 0 < float(x) < 1 else int(x)
        if not 0 <= idx < n:
            raise ValueError(f"слой {idx} вне диапазона 0..{n - 1} (модель имеет {n} слоёв)")
        out.append(idx)
    return sorted(set(out))


def load_model_and_tokenizer(spec: LoadSpec, for_generation: bool = True):
    """Загрузить модель и токенизатор. Возвращает (model, tokenizer, info).

    info идёт в манифест прогона: по нему потом видно, в каком режиме считали.
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

    print(f"[loader] {spec.hf_id}: dtype={dtype} — {why}", flush=True)
    if spec.load_in_4bit:
        print("[loader] ВНИМАНИЕ: 4 бита. Активации отличаются от полноточных, "
              "векторы с квантованной модели несопоставимы с обычными.", flush=True)

    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    model.eval()

    tok = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code, token=os.environ.get("HF_TOKEN")
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    if for_generation:
        tok.padding_side = "left"  # decoder-only: иначе ответы съезжают

    devices = {str(p.device) for p in model.parameters()}
    on_cpu = sum(1 for p in model.parameters() if p.device.type == "cpu")
    if on_cpu and torch.cuda.is_available():
        print(f"[loader] ВНИМАНИЕ: {on_cpu} тензоров на процессоре — не хватило памяти карты. "
              "Задайте max_memory или offload_folder в конфиге.", flush=True)

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
