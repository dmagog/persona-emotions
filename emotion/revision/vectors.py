"""Steering-vector constructions for the revision controls.

All functions are pure (torch in, torch out) and are covered by
``python -m emotion.revision.selftest``.

Notation: ``va`` is the target direction, ``vb`` the subtracted one, both taken
at the published operating layer (index ``layer + 1`` of the saved tensor, the
same off-by-one as ``steer_compose.py``). ``theta`` is the angle between them.

Conditions built here (label suffix in brackets):

* ``a-b`` [raw]          the published ordered difference.
* ``a-r_geo`` [geoK]     main null control. ``r`` has the same norm as ``vb`` and
                         the same projection on ``va``; only the component
                         orthogonal to ``va`` is random. Hence
                         ``||va - r|| == ||va - vb||`` and the shrinkage of ``va``
                         is identical; only the identity of ``b`` differs.
* ``a-r_iso`` [isoK]     the control literally requested by R1 (M1a): an
                         isotropic random vector with ``||r|| == ||vb||``. Note
                         that it is a *stronger* perturbation than ``a-b`` when
                         cos(va, vb) > 0.
* ``lambda*a`` [lam]     "weakened a": the part of ``va - vb`` along ``va``. Tests
                         whether b drops only because a got weaker.
* ``proj`` [proj]        ``va - proj_vb(va)``: removes the vb component from the
                         *intervention vector* (R1 M2, M7b). This is not concept
                         erasure in the residual stream.
* ``centered`` [cent]    ``va - mean(v_1..v_7)`` (R1 M2, optional run P7).
* any of the above rescaled to ``||va||`` [norm] (R1 M3).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from emotion.space import ISEAR_EMOTIONS

VECTOR_FILE = "{emotion}_response_avg_diff.pt"


def load_vectors(vector_dir: Path, layer: int) -> dict[str, torch.Tensor]:
    """Load the seven published directions at ``layer`` (block index), float32 on CPU."""
    out = {}
    for emo in ISEAR_EMOTIONS:
        t = torch.load(Path(vector_dir) / VECTOR_FILE.format(emotion=emo), map_location="cpu")
        out[emo] = t[layer + 1].detach().to(torch.float32).clone()
    return out


def pair_seed(slug: str, a: str, b: str, k: int, kind: str) -> int:
    """Deterministic seed per (model, a, b, replicate, kind). Pre-registered rule."""
    h = hashlib.sha256(f"{slug}|{a}|{b}|{k}|{kind}".encode()).hexdigest()
    return int(h[:8], 16)


def _gaussian(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=g, dtype=torch.float64)


def cos(u: torch.Tensor, v: torch.Tensor) -> float:
    u64, v64 = u.to(torch.float64), v.to(torch.float64)
    return float(u64 @ v64 / (u64.norm() * v64.norm()))


def geo_random(va: torch.Tensor, vb: torch.Tensor, seed: int) -> torch.Tensor:
    """Random r with ||r|| = ||vb|| and <r, va_hat> = <vb, va_hat>."""
    va64, vb64 = va.to(torch.float64), vb.to(torch.float64)
    ua = va64 / va64.norm()
    along = float(vb64 @ ua)                      # ||vb|| cos(theta)
    ortho_norm = float((vb64 - along * ua).norm())  # ||vb|| sin(theta)
    u = _gaussian(va.numel(), seed)
    u = u - (u @ ua) * ua
    u = u / u.norm()
    r = along * ua + ortho_norm * u
    return r.to(torch.float32)


def iso_random(vb: torch.Tensor, seed: int) -> torch.Tensor:
    """Isotropic random r with ||r|| = ||vb||."""
    u = _gaussian(vb.numel(), seed)
    return (u / u.norm() * vb.to(torch.float64).norm()).to(torch.float32)


def lam_a(va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """lambda * va, lambda = 1 - (||vb||/||va||) cos(theta): the va-component of va - vb."""
    va64, vb64 = va.to(torch.float64), vb.to(torch.float64)
    lam = float((va64 - vb64) @ va64 / (va64 @ va64))
    return (lam * va64).to(torch.float32)


def proj_remove(va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    """va minus its projection on vb."""
    va64, vb64 = va.to(torch.float64), vb.to(torch.float64)
    ub = vb64 / vb64.norm()
    return (va64 - (va64 @ ub) * ub).to(torch.float32)


def centered(vecs: dict[str, torch.Tensor], emo: str) -> torch.Tensor:
    mu = torch.stack([vecs[e].to(torch.float64) for e in ISEAR_EMOTIONS]).mean(0)
    return (vecs[emo].to(torch.float64) - mu).to(torch.float32)


def rescale(v: torch.Tensor, target_norm: float) -> torch.Tensor:
    v64 = v.to(torch.float64)
    return (v64 / v64.norm() * target_norm).to(torch.float32)


def norm(v: torch.Tensor) -> float:
    return float(v.to(torch.float64).norm())
