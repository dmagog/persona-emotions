"""Protocol stamps: an artifact is valid only if it was produced by the same settings.

Skipping a stage because its output file exists is a silent recompute in reverse.
The stage sees `steer_specificity_raw.csv`, skips itself, and passes on numbers
produced by a different model, a different token budget, or a different layer.
The numbers themselves cannot reveal this, because the file looks like its own.

A stamp is the sha256 of canonical JSON over the stage name, the protocol
version, the parameters that affect the result, and the stamps of the inputs. It
sits next to the artifact as `<file>.stamp.json` or `<directory>/.stamp.json`.

Three outcomes instead of two:

* match      the stage is skipped, which is what "do not recompute" means;
* mismatch   the stage is neither recomputed nor reused in silence. The chain
             stops and reports which settings differ;
* no stamp   inherited from earlier runs. Also not recomputed, because the
             artifact cost hours. It is reused with a warning and marked
             `unstamped` in the manifest until a person confirms it with
             `--adopt`.

Usage:
    python -m emotion.stamp --show runs/Qwen3-1.7B
    python -m emotion.stamp --adopt runs/Qwen3-1.7B --config configs/models/qwen3-1.7b.yaml
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Bump when the MEANING of a stage changes rather than one of its settings:
# different layer indexing, different pooling, a different prompt set. Every
# earlier artifact then reads as a mismatch, which is the intent.
PROTOCOL_VERSION = 1

STAMP_NAME = ".stamp.json"
SKIP = ("__pycache__", ".DS_Store", STAMP_NAME)

REPO = Path(__file__).resolve().parent.parent


def portable_key(path: Path) -> str:
    """Name an input by its path relative to the repository root.

    An absolute path makes the stamp machine specific. Artifacts are computed on
    the GPU box under `P:\\gpu-tasks\\...`, then live in git and are checked out
    elsewhere. With an absolute key, the first run on another machine stopped the
    chain with a mismatch at every stage: the input had "moved" although nothing
    about it changed except the drive letter. Paths outside the repository stay
    absolute, since nothing promises they are portable.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def stamp_path(artifact: Path) -> Path:
    return artifact / STAMP_NAME if artifact.is_dir() else \
        artifact.with_name(artifact.name + ".stamp.json")


def present(artifact: Path) -> bool:
    """Report whether a result exists. An empty directory is not a result.

    A vector directory survives a crashed run: it exists and holds zero files, so
    a plain "does the path exist" check would skip the stage.
    """
    if artifact.is_dir():
        return any(p.is_file() and p.name != STAMP_NAME for p in artifact.rglob("*"))
    return artifact.exists()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_digest(path: Path) -> str:
    """Digest the content. For a directory, digest file names and contents.

    Content rather than modification time, because artifacts travel between the
    Windows box and a Mac and `scp` without `-p` does not preserve timestamps. An
    mtime check on that route yields either a false "vectors are older than
    pairs", costing hours of regeneration, or false freshness.
    """
    if not path.exists():
        return ""
    if path.is_file():
        return _sha_file(path)
    parts = []
    for p in sorted(path.rglob("*")):
        if not p.is_file() or any(s in p.parts or p.name == s for s in SKIP):
            continue
        if p.name.endswith(".partial.csv"):
            continue  # resume checkpoint, not part of the result
        parts.append(f"{p.relative_to(path).as_posix()}:{_sha_file(p)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def input_digests(inputs: Sequence[Path]) -> dict[str, str]:
    """Digest each input by its stamp when it has one, otherwise by its content.

    The stamp is preferred because it carries a change in a distant input all the
    way down: new pairs give new vectors, and the matrix then reads as a
    mismatch. Content is always appended as well, since an artifact can be edited
    after its stamp was written.
    """
    out: dict[str, str] = {}
    for p in inputs:
        key = portable_key(p)
        prev = read_stamp(p)
        content = content_digest(p)
        out[key] = f"{prev['fingerprint']}+{content[:16]}" if prev else f"content:{content[:16]}"
    return out


def fingerprint(stage: str, params: dict, inputs: Sequence[Path]) -> tuple[str, dict]:
    digs = input_digests(inputs)
    payload = _canon({"stage": stage, "protocol": PROTOCOL_VERSION,
                      "params": params, "inputs": digs})
    return hashlib.sha256(payload.encode()).hexdigest()[:16], digs


def read_stamp(artifact: Path) -> dict | None:
    sp = stamp_path(artifact)
    if not sp.is_file():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_stamp(artifact: Path, stage: str, params: dict,
                inputs: Sequence[Path] = ()) -> dict:
    """Write a stamp next to the artifact. Call this right after a stage succeeds."""
    fp, digs = fingerprint(stage, params, inputs)
    data = {"stage": stage, "protocol": PROTOCOL_VERSION, "fingerprint": fp,
            "params": params, "inputs": digs,
            # Digest of the artifact itself, so the stamp answers "is this the
            # same file" as well as "how was it produced". Without it a manual
            # edit stays invisible until the next stage, by which time the
            # numbers have already reached a report.
            "content": content_digest(artifact),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    stamp_path(artifact).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def diff_params(old: dict, new: dict) -> list[str]:
    """List the differences in words. This text goes into the stop message."""
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, "(none)"), new.get(k, "(none)")
        if a != b:
            out.append(f"{k}: was {a!r}, now {b!r}")
    return out


@dataclass
class Verdict:
    """`state` decides what happens next, `reason` is what the user is told."""
    state: str                       # missing | current | stale | unstamped
    reason: str = ""
    changed: list[str] = field(default_factory=list)

    @property
    def reusable(self) -> bool:
        return self.state in ("current", "unstamped")


def check(artifact: Path, stage: str, params: dict,
          inputs: Sequence[Path] = ()) -> Verdict:
    if not present(artifact):
        return Verdict("missing", "no artifact")
    prev = read_stamp(artifact)
    if prev is None:
        return Verdict("unstamped",
                       "artifact has no stamp: produced before stamps existed, "
                       "or copied from another machine")
    fp, digs = fingerprint(stage, params, inputs)
    now = content_digest(artifact)
    if prev.get("fingerprint") == fp and prev.get("content", now) == now:
        return Verdict("current", "stamp matches")

    changed = []
    if prev.get("content", now) != now:
        changed.append("the artifact itself changed after its stamp was written")
    if prev.get("protocol") != PROTOCOL_VERSION:
        changed.append(f"protocol version: was {prev.get('protocol')}, "
                       f"now {PROTOCOL_VERSION}")
    changed += diff_params(prev.get("params") or {}, params)
    for k in sorted(set(prev.get("inputs") or {}) | set(digs)):
        a, b = (prev.get("inputs") or {}).get(k), digs.get(k)
        if a != b:
            changed.append(f"input {k}: recomputed or edited")
    if not changed:
        changed.append("the stamp differs although the settings match; "
                       "check the protocol version")
    return Verdict("stale", "artifact was produced with different settings", changed)


def decide(artifact: Path, stage: str, params: dict, inputs: Sequence[Path] = (),
           recompute_stale: bool = False, label: str | None = None) -> bool:
    """Print the verdict and return True when the stage has to run.

    A mismatched artifact stops the chain by default. Recomputing it silently
    costs a night of generation; reusing it silently puts a row produced under
    another protocol into a report. The choice belongs to a person, through
    `--recompute-stale` or by moving the file away.
    """
    name = label or artifact.name
    v = check(artifact, stage, params, inputs)
    if v.state == "current":
        print(f"   {name}: stamp matches, skipping", flush=True)
        return False
    if v.state == "unstamped":
        print(f"   {name}: WARNING, no stamp. Reusing as is.\n"
              f"      If it was produced by the current protocol, confirm it:\n"
              f"      python -m emotion.stamp --adopt {artifact.parent}", flush=True)
        return False
    if v.state == "stale":
        detail = "\n".join(f"      - {c}" for c in v.changed)
        if recompute_stale:
            print(f"   {name}: settings differ, recomputing\n{detail}", flush=True)
            return True
        raise SystemExit(
            f"\n{name}: this artifact was produced with different settings.\n{detail}\n\n"
            f"  It cost time, so the chain will not touch it on its own. Choose one:\n"
            f"    - recompute  run again with --recompute-stale\n"
            f"    - keep       rename {artifact} and run again\n"
            f"    - adopt      python -m emotion.stamp --adopt {artifact.parent}\n"
        )
    return True


# --- adoption and inspection -------------------------------------------------

def adopt(artifact: Path, stage: str, params: dict,
          inputs: Sequence[Path] = ()) -> dict | None:
    """Stamp an existing artifact with the current protocol.

    A deliberate human statement: yes, this file was produced by these settings.
    It exists for runs computed before stamps were introduced, so that four and a
    half hours are not spent again for the sake of one signature.
    """
    if not present(artifact):
        return None
    return write_stamp(artifact, stage, params, inputs)


def show(run_dir: Path) -> None:
    """Report how each artifact of a run was produced. Start here when a row looks wrong."""
    found = False
    for sp in sorted({*run_dir.rglob("*stamp.json"), *run_dir.rglob(STAMP_NAME)}):
        try:
            d = json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        found = True
        target = sp.parent.name if sp.name == STAMP_NAME else sp.name[:-len(".stamp.json")]
        print(f"\n{target}  [{d.get('stage')}] {d.get('fingerprint')}  "
              f"protocol {d.get('protocol')}  {d.get('created', '')}")
        for k, v in sorted((d.get("params") or {}).items()):
            print(f"    {k}: {v}")
        for k, v in sorted((d.get("inputs") or {}).items()):
            print(f"    <- {k}  {v}")
    if not found:
        print(f"{run_dir}: no stamps, this run predates them")


def verify(run_dir: Path) -> tuple[int, list[str]]:
    """Check that artifact contents still match their stamps.

    Needed after a transfer. Artifacts travel from the GPU box to a Mac, and an
    interrupted copy leaves a truncated CSV that still opens and parses, only
    with fewer rows. A stamp catches that; the eye does not.
    """
    bad: list[str] = []
    total = 0
    for sp in sorted({*run_dir.rglob("*stamp.json"), *run_dir.rglob(STAMP_NAME)}):
        total += 1
        try:
            d = json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            bad.append(f"{sp.name}: stamp is unreadable")
            continue
        art = sp.parent if sp.name == STAMP_NAME else \
            sp.with_name(sp.name[:-len(".stamp.json")])
        if not present(art):
            bad.append(f"{art.name}: stamp exists, artifact does not")
            continue
        want = d.get("content")
        if want is None:
            continue  # older stamp, written before content digests
        if content_digest(art) != want:
            bad.append(f"{art.name}: content no longer matches the stamp "
                       "(interrupted copy or a manual edit)")
    return total, bad


def unstamped(paths: Iterable[Path]) -> list[str]:
    """List artifacts without a stamp. The manifest records them and the summary flags them."""
    return [p.name for p in paths if present(p) and read_stamp(p) is None]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Protocol stamps for the artifacts of a run.")
    ap.add_argument("run", type=Path, help="a runs/<slug> directory")
    ap.add_argument("--show", action="store_true", help="print the stamps (default)")
    ap.add_argument("--adopt", action="store_true",
                    help="stamp existing artifacts with the current config")
    ap.add_argument("--verify", action="store_true",
                    help="check artifact contents against their stamps, after a transfer")
    ap.add_argument("--config", type=Path, default=None,
                    help="model config, required by --adopt")
    args = ap.parse_args()

    if args.verify:
        total, bad = verify(args.run)
        for b in bad:
            print(f"  FAIL {b}")
        if bad:
            raise SystemExit(f"{args.run}: {len(bad)} of {total} artifacts did not survive the copy")
        if total == 0:
            # Zero stamps is not "everything matches", it is "nothing to check".
            # This directory used to report success and exit 0, so a transfer
            # check went green on a run that had no stamps at all.
            raise SystemExit(
                f"{args.run}: no stamps, nothing to check. If this run was produced by "
                f"the current protocol, confirm it: python3 -m emotion.stamp {args.run} "
                f"--adopt --config configs/models/<config>.yaml")
        print(f"{args.run}: checked {total} artifacts, all match their stamps")
        return

    if args.adopt:
        from emotion.run_model_chain import stage_specs
        if not args.config:
            raise SystemExit("--adopt requires --config: the stamp has to record "
                             "which settings produced the artifact")
        n = 0
        for spec in stage_specs(args.config):
            if adopt(spec.artifact, spec.stage, spec.params, spec.inputs):
                print(f"  stamped {spec.artifact}")
                n += 1
        print(f"artifacts adopted: {n}")
        return

    show(args.run)


if __name__ == "__main__":
    main()
