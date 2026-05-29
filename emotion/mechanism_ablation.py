"""Why does SAE help - sparse features or just removing the shared axis? (Phase 3.4)

SAE contrastive features disentangle at the FEATURE-INDEX level (off-diag Jaccard 0.25
vs 0.91 for projection). But the steering vector is those features DECODED back into
the residual stream (sum delta*W_dec[f], renormed). Question: in residual space, is the
SAE-feature vector more disentangled than the raw vector - and is it more than what you
get by simply subtracting the shared mean axis ("raw minus common component")?

Compares, at one layer, the pairwise-cosine entanglement of three vector sets:
  1. raw emotion vectors;
  2. centered raw (raw - mean over the 7) = "remove the shared axis";
  3. SAE-feature vectors (decoded top contrastive features).
Plus how much each set still aligns with the raw shared-mean axis. No GPU.

Usage:
    python -m emotion.mechanism_ablation \
        --raw-dir emotion_vectors/gemma-2-2b-it \
        --sae-dir emotion_vectors/gemma-sae-feat --layer 12 \
        --contrastive-json results/sae_contrastive_layer12.json \
        --out results/mechanism_ablation_gemma.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from emotion.space import ISEAR_EMOTIONS


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-8))


def offdiag_mean(vecs):
    es = list(vecs)
    cs = [cos(vecs[a], vecs[b]) for i, a in enumerate(es) for j, b in enumerate(es) if j > i]
    return sum(cs) / len(cs), min(cs), max(cs)


def load_set(d: Path, layer: int):
    return {e: torch.load(d / f"{e}_response_avg_diff.pt", map_location="cpu")[layer + 1].float()
            for e in ISEAR_EMOTIONS}


def main() -> None:
    ap = argparse.ArgumentParser(description="Mechanism ablation: SAE features vs centering.")
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--sae-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--contrastive-json", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    raw = load_set(args.raw_dir, args.layer)
    sae = load_set(args.sae_dir, args.layer)
    mean_raw = torch.stack([raw[e] for e in ISEAR_EMOTIONS]).mean(0)
    cen = {e: raw[e] - mean_raw for e in ISEAR_EMOTIONS}

    out = []
    def emit(s=""):
        print(s); out.append(s)

    emit(f"# Механизм: SAE-фичи vs центрирование (Phase 3.4) — layer {args.layer}\n")

    emit("## Запутанность направлений (средний попарный косинус, off-diag)\n")
    emit("| набор векторов | mean cos | min | max |")
    emit("|---|---:|---:|---:|")
    for name, vs in [("raw (сырые)", raw), ("centered (raw − общая ось)", cen), ("SAE-фичи (декод)", sae)]:
        m, lo, hi = offdiag_mean(vs)
        emit(f"| {name} | {m:+.2f} | {lo:+.2f} | {hi:+.2f} |")

    emit("\n## Остаток общей оси (косинус с raw-mean — «эмоциональность»)\n")
    emit("| набор | mean |cos с общей осью| |")
    emit("|---|---:|")
    for name, vs in [("raw", raw), ("centered", cen), ("SAE-фичи", sae)]:
        a = sum(abs(cos(vs[e], mean_raw)) for e in ISEAR_EMOTIONS) / len(ISEAR_EMOTIONS)
        emit(f"| {name} | {a:.2f} |")

    if args.contrastive_json and args.contrastive_json.exists():
        cj = json.loads(args.contrastive_json.read_text())
        jac = cj.get("mean_offdiag_jaccard")
        emit(f"\nДля контекста: на уровне НАБОРА фич (индексы) off-diag Jaccard = {jac:.2f} "
             "(против 0.91 у проекции) — фичи разные. Вопрос — переживает ли это декод в residual.")

    emit("\n## Вывод\n")
    mr, _, _ = offdiag_mean(raw)
    mc, _, _ = offdiag_mean(cen)
    ms, _, _ = offdiag_mean(sae)
    emit(f"- raw cos {mr:+.2f} → centered {mc:+.2f} → SAE-фичи {ms:+.2f}.")
    if ms < mc - 0.05:
        emit("- SAE-фичи расщепляют направления СИЛЬНЕЕ простого центрирования — значит дело "
             "не только в удалении общей оси, а в разреженном отборе.")
    elif abs(ms - mc) <= 0.05:
        emit("- SAE-фичи дают примерно то же расщепление, что и центрирование — основная часть "
             "выигрыша SAE в residual-пространстве ≈ удаление общей оси (а не сам по себе отбор).")
    else:
        emit("- SAE-фичи в residual-пространстве остаются БОЛЕЕ запутанными, чем центрированные "
             "(и почти как сырые): расщепление на уровне индексов фич плохо переживает декод "
             "(строки W_dec сами скоррелированы, общие фичи доминируют в сумме). Согласуется с "
             "W2 (нет поведенческого выигрыша по специфичности).")
        emit("- При этом центрирование даёт низкий косинус ценой пере-коррекции (§5: схлопывает "
             "sadness) — низкий cos там не «бесплатный». Итог: разреженность на уровне фич ≠ "
             "ортогональность направлений стиринга; вот где узкое место для специфичности.")

    if args.out is not None:
        args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
