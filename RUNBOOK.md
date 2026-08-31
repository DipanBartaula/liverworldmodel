# Reviewer Runbook

## What This Package Is

This is a self-contained evidence package for eight selected liver world-model experiments. The
primary submission is [DECISION_MEMO.md](DECISION_MEMO.md); the README supplies detailed tables,
plots, architecture diagram, exact source map, and raw run artifacts.

## Included

- Synthetic datasets used by the evaluation harness: `data/liver_data.csv` and `data/liver_data_long.csv`.
- Exact selected-eight source, training, inference, evaluation, and constraint code under `code/`.
- Raw sweep results, selected best-run JSONs, available deep metrics, and a package manifest.
- Generated report plots and a no-dependency verifier.

## Deliberately Not Claimed As Included

The package does **not** contain the trained RateAnchor checkpoint used for the headline result.
The recorded best-run metrics and exact rebuild configuration are included, but a reviewer cannot
reproduce the exact saved weights without retraining. This is intentional disclosure, not a hidden
dependency. The month-30 patient-level explanation described in the memo is therefore a stated next
artifact, not a fabricated result.

## Setup

Use Python 3.10-3.12. Create a virtual environment, then install the reference dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CPU-only PyTorch, the exact wheel availability depends on your platform. If the `+cpu` wheel is
not resolved by your index, install the matching CPU PyTorch wheel from the official PyTorch index,
then rerun `pip install -r requirements.txt`.

## First Command: Verify The Handoff

```powershell
python scripts/verify_package.py
```

The verifier checks data checksums, the eight-model manifest, architecture source mapping, artifact
coverage, generated visuals, and the decision memo. It does not retrain models or claim a checkpoint
is present.

## Rebuild A Selected Configuration

The precise selected scale and epoch count appear in `artifacts/selected8_manifest.json`. For example:

```powershell
python code/training.py --model rate_anchor --epochs 15 --scale 4.854166666727
python code/eval.py --model rate_anchor --ckpt checkpoints/rate_anchor.pt --csv data/liver_data.csv --csv-long data/liver_data_long.csv
python code/inference.py --model rate_anchor --ckpt checkpoints/rate_anchor.pt --csv data/liver_data.csv --patient 0 --observe 24
```

## Rebuild The Evidence Figures

```powershell
python scripts/build_selected8_package.py
```

This refreshes copied artifacts, manifests, plots, and README from the parent experiment workspace.
It is a maintainer command, not required by an external reviewer using this package.
