# Selected Eight: Code Map

This package keeps the common executable pipeline and the architecture-specific source for the
eight experiments used in the paper. `models/` also includes a few transitive dependencies so the
selected architectures can be imported without hidden parent-repository files.

## Common Pipeline

| Concern | File | Role |
|---|---|---|
| Synthetic liver trajectories | `Data_Generator.py` | Seeded generator and disease dynamics. |
| Dataset / OOD splits | `dataloader.py` | Train/validation data, susceptibility, treatment-timing, and long-horizon probes. |
| Training loop | `training.py` | Model construction, optimization, checkpoints, and training entry point. |
| Scale sweep | `sweep.py` | Parameter-budget sweep used for scaling results. |
| Evaluation | `eval.py` | Free rollouts, MAE, constraints, OOD probes, horizon probe, and effective-rank check. |
| Patient inference | `inference.py` | Predicted-versus-true trajectory inspection for a selected patient. |
| Constraints / metrics | `util.py` | `ConstraintHead`, state loss, effective rank, and violation counter. |

## Architecture-Specific Files

| Experiment | Primary architecture source | Required local architecture dependencies |
|---|---|---|
| `simple_transformer` | `models/simple_transformer.py` | `models/base.py` |
| `ts_jepa` | `models/ts_jepa.py` | `models/base.py`, `models/le_world_model.py` (`vicreg`) |
| `ada_jepa` | `models/ada_jepa.py` | `models/base.py`, `models/composed.py`, `models/le_world_model.py`, `models/mamba2.py` |
| `multihorizon_meta_le_world_model` | `models/multihorizon_meta_le_world_model.py` | `models/base.py` |
| `gnn_leworld_meta_pde` | `models/neural_pde.py` | `models/base.py`, `models/neural_sde.py`, `models/composed.py` |
| `gnn_leworld_meta_ode` | `models/neural_pde.py` | `models/base.py`, `models/neural_sde.py`, `models/composed.py` |
| `rate_anchor` | `models/rate_family.py` | `models/neural_pde.py`, `models/composed.py`, `models/base.py`, `models/neural_sde.py` |
| `rate_anchor_ac` | `models/anticollapse.py` | `models/rate_family.py`, `models/le_world_model.py`, `models/neural_pde.py`, `models/composed.py` |

## Rebuild Pattern

Run from the package root after placing the synthetic CSV files and desired checkpoint location in
the expected paths:

```powershell
python code/training.py --model rate_anchor --epochs 15 --scale 4.854166666727
python code/eval.py --model rate_anchor --ckpt checkpoints/rate_anchor.pt
python code/inference.py --model rate_anchor --ckpt checkpoints/rate_anchor.pt --patient 0 --observe 24
```

The exact chosen parameter scale and best-run JSON for every selected model are listed in the
package `README.md` and `artifacts/selected8_manifest.json`.
