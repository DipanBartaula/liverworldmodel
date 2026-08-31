"""Fail-fast integrity checks for the selected-eight take-home handoff."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "code" / "models"

EXPECTED_MODELS = {
    "simple_transformer": ["simple_transformer.py", "base.py"],
    "ts_jepa": ["ts_jepa.py", "base.py", "le_world_model.py"],
    "ada_jepa": ["ada_jepa.py", "base.py", "composed.py", "le_world_model.py", "mamba2.py"],
    "multihorizon_meta_le_world_model": ["multihorizon_meta_le_world_model.py", "base.py"],
    "gnn_leworld_meta_pde": ["neural_pde.py", "base.py", "neural_sde.py", "composed.py"],
    "gnn_leworld_meta_ode": ["neural_pde.py", "base.py", "neural_sde.py", "composed.py"],
    "rate_anchor": ["rate_family.py", "neural_pde.py", "composed.py", "base.py", "neural_sde.py"],
    "rate_anchor_ac": ["anticollapse.py", "rate_family.py", "le_world_model.py", "neural_pde.py", "composed.py"],
}

REQUIRED_ROOT = ["README.md", "DECISION_MEMO.md", "RUNBOOK.md", "requirements.txt"]
REQUIRED_CODE = [
    "Data_Generator.py", "dataloader.py", "training.py", "sweep.py", "eval.py", "inference.py",
    "util.py", "SELECTED_EIGHT_CODE_MAP.md",
]
REQUIRED_PLOTS = [
    "story_ratchet_mae.png", "story_ood_probes.png", "story_scaling_curves.png",
    "story_rank_vs_accuracy.png", "story_baseline_delta.png", "rate_anchor_architecture.png",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str, errors: list[str]) -> None:
    marker = "OK" if condition else "MISSING/INVALID"
    print(f"[{marker}] {message}")
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []
    for name in REQUIRED_ROOT:
        check((ROOT / name).is_file(), name, errors)
    for name in REQUIRED_CODE:
        check((ROOT / "code" / name).is_file(), f"code/{name}", errors)

    manifest_path = ROOT / "artifacts" / "selected8_manifest.json"
    check(manifest_path.is_file(), "artifacts/selected8_manifest.json", errors)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    experiments = manifest.get("experiments", [])
    names = [item.get("model") for item in experiments]
    check(names == list(EXPECTED_MODELS), "manifest contains exactly the ordered selected eight", errors)

    for model, sources in EXPECTED_MODELS.items():
        check((ROOT / "artifacts" / "best_runs" / f"{model}.json").is_file(), f"best-run artifact: {model}", errors)
        check((ROOT / "artifacts" / "sweeps" / model).is_dir(), f"full sweep artifacts: {model}", errors)
        for source in sources:
            check((MODELS_DIR / source).is_file(), f"architecture source: {model} -> {source}", errors)

    for name in REQUIRED_PLOTS:
        check((ROOT / "plots" / name).is_file(), f"plot: {name}", errors)

    data_manifest_path = DATA_DIR / "manifest.json"
    check(data_manifest_path.is_file(), "data/manifest.json", errors)
    if data_manifest_path.exists():
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        for item in data_manifest.get("files", []):
            path = DATA_DIR / item["name"]
            check(path.is_file(), f"data file: {item['name']}", errors)
            if path.exists():
                check(path.stat().st_size == item["bytes"], f"data size: {item['name']}", errors)
                check(sha256(path) == item["sha256"], f"data checksum: {item['name']}", errors)

    if errors:
        print(f"\nVerification failed with {len(errors)} issue(s).")
        return 1
    print("\nVerification passed: selected-eight handoff is internally consistent.")
    print("Note: RUNBOOK.md documents the intentionally absent final RateAnchor checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
