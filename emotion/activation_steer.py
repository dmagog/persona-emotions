# activation_steering.py  – v0.2
import torch
from contextlib import contextmanager
from typing import Sequence, Union, Iterable


def _hidden_size(config):
    """hidden_size, falling back to the nested config that multimodal models use."""
    h = getattr(config, "hidden_size", None)
    if isinstance(h, int):
        return h
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            sub = getter()
            h = getattr(sub, "hidden_size", None)
            if isinstance(h, int):
                return h
        except Exception:
            pass
    for attr in ("text_config", "llm_config", "language_config"):
        sub = getattr(config, attr, None)
        h = getattr(sub, "hidden_size", None) if sub is not None else None
        if isinstance(h, int):
            return h
    return None


class ActivationSteerer:
    """
    Add (coeff * steering_vector) to a chosen transformer block's output.
    Now handles blocks that return tuples and fails loudly if it can't
    locate a layer list.
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "model.layers",                      # Llama, Mistral, Qwen, Gemma-2
        "model.language_model.layers",       # Gemma-3, Llama-4, multimodal
        "language_model.model.layers",       # some VL wrappers
        "model.language_model.model.layers",
        "transformer.h",                     # GPT-2/Neo, Bloom
        "gpt_neox.layers",                   # GPT-NeoX
        "encoder.layer",                     # BERT/RoBERTa
        "encoder.block",                     # T5, which has no model.block
    )

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        debug: bool = False,
    ):
        self.model, self.coeff, self.layer_idx = model, float(coeff), layer_idx
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None

        # --- build vector ---
        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1‑D")
        # Multimodal models keep hidden_size in a nested config. Without the
        # fallback getattr returns None and the shape check silently turns off.
        hidden = _hidden_size(model.config)
        if hidden is None:
            raise ValueError(
                "hidden_size could not be read from the model config, so the vector "
                "shape cannot be checked, and without that check it is easy to steer "
                "with garbage"
            )
        if self.vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
            )
        if not torch.isfinite(self.vector).all():
            raise ValueError(
                "the steering vector holds inf or nan, which usually means empty "
                "responses during extraction; steering with it produces garbage"
            )
        if float(self.vector.norm()) <= 1e-6:
            raise ValueError(
                f"zero steering vector (norm {float(self.vector.norm()):.2e}). "
                "SAE vectors are usually non-zero on one layer only, so check that "
                "the layer matches the one the vector was built on"
            )
        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")

    # ---------- helpers ----------
    def _locate_layer(self):
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:  # found a full match
                if not hasattr(cur, "__getitem__"):
                    continue  # not a list/ModuleList
                if not (-len(cur) <= self.layer_idx < len(cur)):
                    raise IndexError("layer_idx out of range")
                if self.debug:
                    print(f"[ActivationSteerer] hooking {path}[{self.layer_idx}]")
                return cur[self.layer_idx]

        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )

    def _hook_fn(self, module, ins, out):
        steer = self.coeff * self.vector  # (hidden,)

        def _add(t):
            if self.positions == "all":
                return t + steer.to(t.device)
            elif self.positions == "prompt":
                if t.shape[1] == 1:
                    return t
                else:
                    t2 = t.clone()
                    t2 += steer.to(t.device)
                    return t2
            elif self.positions == "response": 
                t2 = t.clone()
                t2[:, -1, :] += steer.to(t.device)
                return t2
            else:
                raise ValueError(f"Invalid positions: {self.positions}")

        # out may be tensor or tuple/list => normalise to tuple
        if torch.is_tensor(out):
            new_out = _add(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                # unusual case – don't touch
                return out
            head = _add(out[0])
            new_out = (head, *out[1:])  # keep other entries
        else:
            return out  # unknown type – leave unchanged

        if self.debug:
            with torch.no_grad():
                delta = (new_out[0] if isinstance(new_out, tuple) else new_out) - (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                print(
                    "[ActivationSteerer] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_out

    # ---------- context manager ----------
    def __enter__(self):
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        self.remove()  # always clean up

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


class ActivationSteererMultiple:
    """
    Add multiple (coeff * steering_vector) to chosen transformer block outputs.
    Accepts a list of dicts, each with keys: steering_vector, coeff, layer_idx, positions.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict],
        *,
        debug: bool = False,
    ):
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._handles = []
        self._steerers = []

        # Validate and create individual steerers
        for inst in self.instructions:
            steerer = ActivationSteerer(
                model,
                inst["steering_vector"],
                coeff=inst.get("coeff", 0.0),
                layer_idx=inst.get("layer_idx", -1),
                positions=inst.get("positions", "all"),
                debug=debug,
            )
            self._steerers.append(steerer)

    def __enter__(self):
        for steerer in self._steerers:
            layer = steerer._locate_layer()
            handle = layer.register_forward_hook(steerer._hook_fn)
            steerer._handle = handle
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for steerer in self._steerers:
            steerer.remove()
        self._handles.clear()