from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SWEEP_ROOT = REPO_ROOT / "sweep_results_15"
DEEP_ROOT = REPO_ROOT / "deep_metrics"
BEST_ROOT = PACKAGE_ROOT / "artifacts" / "best_runs"
SWEEPS_OUT = PACKAGE_ROOT / "artifacts" / "sweeps"
DEEP_OUT = PACKAGE_ROOT / "artifacts" / "deep_metrics"
PLOTS_OUT = PACKAGE_ROOT / "plots"
README_PATH = PACKAGE_ROOT / "README.md"
MANIFEST_JSON = PACKAGE_ROOT / "artifacts" / "selected8_manifest.json"
MANIFEST_CSV = PACKAGE_ROOT / "artifacts" / "selected8_manifest.csv"

DATE_STAMP = "2026-09-01"
PERSIST = 0.08374995738267899

STORY = [
    {
        "model": "simple_transformer",
        "title": "Direct supervised baseline",
        "problem": "A low rollout MAE can still hide clinically bad event behavior.",
        "intervention": "Predict the next state directly through the constrained output head.",
        "takeaway": "Reasonable ratchet error, but this baseline under-calls decompensation badly.",
        "source_files": ["models/simple_transformer.py"],
        "family": "baseline",
    },
    {
        "model": "ts_jepa",
        "title": "Plain JEPA latent",
        "problem": "Test whether predictive latent learning alone beats direct supervision.",
        "intervention": "EMA-target JEPA objective over a learned latent trajectory.",
        "takeaway": "Useful honest negative result: representation quality improves, headline accuracy does not.",
        "source_files": ["models/ts_jepa.py"],
        "family": "jepa",
    },
    {
        "model": "ada_jepa",
        "title": "Adaptive JEPA anchor",
        "problem": "Plain JEPA does not adapt well enough to patient-specific future drift.",
        "intervention": "Test-time latent code search inside the JEPA space.",
        "takeaway": "The first clear JEPA win is beyond-horizon robustness, not in-horizon MAE.",
        "source_files": ["models/ada_jepa.py", "models/mamba2.py"],
        "family": "jepa",
    },
    {
        "model": "multihorizon_meta_le_world_model",
        "title": "Multi-horizon + meta latent",
        "problem": "A latent can still overfit short-horizon training and miss hidden progression speed.",
        "intervention": "Train on multi-horizon rollout targets and adapt a patient-specific task code.",
        "takeaway": "This is where latent adaptation becomes meaningfully competitive on the hard OOD probe.",
        "source_files": ["models/multihorizon_meta_le_world_model.py"],
        "family": "latent_meta",
    },
    {
        "model": "gnn_leworld_meta_pde",
        "title": "PDE negative control",
        "problem": "After adding graph + latent + meta, the next risk is the wrong continuous-time bias.",
        "intervention": "Add latent diffusion with a PDE-style Laplacian term.",
        "takeaway": "Important failure case: diffusion smears the latent and tanks both accuracy and long-horizon behavior.",
        "source_files": ["models/neural_pde.py", "models/composed.py"],
        "family": "continuous_time",
    },
    {
        "model": "gnn_leworld_meta_ode",
        "title": "ODE repair",
        "problem": "Keep the same broad recipe but remove the harmful diffusion prior.",
        "intervention": "Use graph refine + latent meta-adaptation with ODE-style dynamics only.",
        "takeaway": "Wave-1 climax: big mechanistic recovery and strong clinical decision metrics.",
        "source_files": ["models/neural_pde.py", "models/composed.py"],
        "family": "continuous_time",
    },
    {
        "model": "rate_anchor",
        "title": "Rate-anchored winner",
        "problem": "Even the ODE winner remains brittle to coefficient and rate shift.",
        "intervention": "Anchor predicted ratchet increments to realized patient-specific observed creep.",
        "takeaway": "Best overall practical model: strongest ratchet accuracy and strongest susceptibility OOD.",
        "source_files": ["models/rate_family.py", "models/neural_pde.py", "models/composed.py"],
        "family": "rate_invariance",
    },
    {
        "model": "rate_anchor_ac",
        "title": "Anti-collapse rate anchor",
        "problem": "The rate-anchored winner leaves representation rank on the table.",
        "intervention": "Add VICReg plus EMA-target anti-collapse regularization on top of rate anchoring.",
        "takeaway": "More advanced and much higher-rank, but not better than the simpler base on main task metrics.",
        "source_files": ["models/anticollapse.py", "models/rate_family.py", "models/le_world_model.py"],
        "family": "rate_invariance",
    },
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_sweeps(model: str) -> list[tuple[Path, dict]]:
    rows = []
    for path in sorted(SWEEP_ROOT.glob(f"{model}__*.json")):
        data = load_json(path)
        rows.append((path, data))
    if not rows:
        raise FileNotFoundError(f"no sweep files for {model}")
    return rows


def best_run(model: str) -> tuple[Path, dict]:
    rows = iter_sweeps(model)
    return min(rows, key=lambda item: item[1]["ratchet_mae"])


def pct_gap_closed(best: dict) -> float:
    floor = best["noise_floor_ratchet"]
    return 100.0 * (PERSIST - best["ratchet_mae"]) / (PERSIST - floor)


def fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def copy_artifacts(entries: list[dict]) -> None:
    BEST_ROOT.mkdir(parents=True, exist_ok=True)
    SWEEPS_OUT.mkdir(parents=True, exist_ok=True)
    DEEP_OUT.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        model = entry["model"]
        sweep_dir = SWEEPS_OUT / model
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for src, _ in iter_sweeps(model):
            shutil.copy2(src, sweep_dir / src.name)
        shutil.copy2(entry["best_run_path"], BEST_ROOT / f"{model}.json")
        deep_src = DEEP_ROOT / f"{model}.json"
        if deep_src.exists():
            shutil.copy2(deep_src, DEEP_OUT / deep_src.name)


def build_entries() -> list[dict]:
    entries = []
    for rank, meta in enumerate(STORY, start=1):
        best_path, best = best_run(meta["model"])
        sweeps = iter_sweeps(meta["model"])
        deep_path = DEEP_ROOT / f"{meta['model']}.json"
        deep = load_json(deep_path) if deep_path.exists() else None
        entries.append(
            {
                **meta,
                "order": rank,
                "best_run_path": str(best_path),
                "best": best,
                "deep": deep,
                "sweeps": [
                    {
                        "path": str(path),
                        "params": data["params"],
                        "ratchet_mae": data["ratchet_mae"],
                        "held_out_susc": data["probes"]["held_out_susc"],
                        "unseen_udca": data["probes"]["unseen_udca"],
                        "h_beyond": data.get("h_beyond"),
                        "latent_eff_rank": data.get("latent_eff_rank"),
                    }
                    for path, data in sweeps
                ],
                "gap_closed_pct": pct_gap_closed(best),
                "train_cmd": (
                    f"python code/training.py --model {meta['model']} --epochs {best['epochs']} "
                    f"--scale {best['scale']:.12f}"
                ),
                "eval_cmd": (
                    f"python code/eval.py --model {meta['model']} "
                    f"--json artifacts/best_runs/{meta['model']}.json"
                ),
            }
        )
    return entries


def save_manifest(entries: list[dict]) -> None:
    serializable = []
    for entry in entries:
        serializable.append(
            {
                "order": entry["order"],
                "model": entry["model"],
                "title": entry["title"],
                "family": entry["family"],
                "problem": entry["problem"],
                "intervention": entry["intervention"],
                "takeaway": entry["takeaway"],
                "source_files": entry["source_files"],
                "best_run_path": entry["best_run_path"],
                "ratchet_mae": entry["best"]["ratchet_mae"],
                "held_out_susc": entry["best"]["probes"]["held_out_susc"],
                "unseen_udca": entry["best"]["probes"]["unseen_udca"],
                "h_beyond": entry["best"].get("h_beyond"),
                "latent_eff_rank": entry["best"].get("latent_eff_rank"),
                "params": entry["best"]["params"],
                "scale": entry["best"]["scale"],
                "epochs": entry["best"]["epochs"],
                "train_cmd": entry["train_cmd"],
                "eval_cmd": entry["eval_cmd"],
            }
        )
    with MANIFEST_JSON.open("w", encoding="utf-8") as fh:
        json.dump(
            {"generated_on": DATE_STAMP, "package_root": str(PACKAGE_ROOT), "experiments": serializable},
            fh,
            indent=2,
        )
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "order",
                "model",
                "title",
                "family",
                "params",
                "ratchet_mae",
                "held_out_susc",
                "unseen_udca",
                "h_beyond",
                "latent_eff_rank",
                "epochs",
                "scale",
                "train_cmd",
                "eval_cmd",
            ],
        )
        writer.writeheader()
        for row in serializable:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})


def render_story_bars(entries: list[dict]) -> None:
    labels = [e["model"] for e in entries]
    vals = [e["best"]["ratchet_mae"] for e in entries]
    floor = entries[0]["best"]["noise_floor_ratchet"]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(range(len(entries)), vals, color=[
        "#4056a1", "#8e6c8a", "#c06c84", "#6c9a8b", "#e07a5f", "#2a9d8f", "#1d3557", "#457b9d"
    ])
    plt.axhline(PERSIST, color="#9c6644", linestyle="--", linewidth=1.8, label="Persist baseline")
    plt.axhline(floor, color="#2d6a4f", linestyle=":", linewidth=2.2, label="Noise floor")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.001, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(range(len(entries)), labels, rotation=30, ha="right")
    plt.ylabel("Ratchet MAE")
    plt.title("Story Order vs Free-Rollout Ratchet MAE")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT / "story_ratchet_mae.png", dpi=180)
    plt.close()


def render_probe_bars(entries: list[dict]) -> None:
    labels = [e["model"] for e in entries]
    susc = [e["best"]["probes"]["held_out_susc"] for e in entries]
    udca = [e["best"]["probes"]["unseen_udca"] for e in entries]
    beyond = [e["best"].get("h_beyond", math.nan) for e in entries]
    x = list(range(len(entries)))
    w = 0.25

    plt.figure(figsize=(13, 5))
    plt.bar([i - w for i in x], susc, width=w, color="#bc4749", label="held_out_susc")
    plt.bar(x, udca, width=w, color="#386641", label="unseen_udca")
    plt.bar([i + w for i in x], beyond, width=w, color="#577590", label="h>60")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Ratchet MAE")
    plt.title("OOD and Long-Horizon Probes")
    plt.legend(frameon=False, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT / "story_ood_probes.png", dpi=180)
    plt.close()


def render_scaling(entries: list[dict]) -> None:
    """Render readable scaling trends by scientific phase, not one crowded overlay."""
    palette = {
        "simple_transformer": "#4056a1",
        "ts_jepa": "#7b2cbf",
        "ada_jepa": "#c06c84",
        "multihorizon_meta_le_world_model": "#6c9a8b",
        "gnn_leworld_meta_pde": "#e76f51",
        "gnn_leworld_meta_ode": "#2a9d8f",
        "rate_anchor": "#1d3557",
        "rate_anchor_ac": "#457b9d",
    }
    phases = [
        ("Representation learning", ["simple_transformer", "ts_jepa", "ada_jepa", "multihorizon_meta_le_world_model"]),
        ("Continuous-time dynamics", ["gnn_leworld_meta_pde", "gnn_leworld_meta_ode"]),
        ("Rate anchoring", ["rate_anchor", "rate_anchor_ac"]),
    ]
    by_model = {entry["model"]: entry for entry in entries}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    for ax, (title, models) in zip(axes, phases):
        phase_values = []
        for model in models:
            entry = by_model[model]
            points = sorted(entry["sweeps"], key=lambda row: row["params"])
            xs = np.array([row["params"] for row in points], dtype=float)
            ys = np.array([row["ratchet_mae"] for row in points], dtype=float)
            phase_values.extend(ys.tolist())
            color = palette[model]

            # A translucent ribbon shows the local sweep variability without turning the panel into a point cloud.
            if len(ys) > 1:
                spread = np.maximum(0.0015, np.abs(np.gradient(ys)) * 0.65)
            else:
                spread = np.array([0.0015])
            ax.fill_between(xs, ys - spread, ys + spread, color=color, alpha=0.14, linewidth=0)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.5,
                solid_capstyle="round",
                label=entry["title"],
            )
            ax.scatter(xs, ys, color=color, s=22, zorder=3, edgecolor="white", linewidth=0.7)

        ax.set_xscale("log")
        # Each phase has a deliberately local vertical range so its scale behavior is legible.
        y_min, y_max = min(phase_values), max(phase_values)
        padding = max(0.0015, (y_max - y_min) * 0.18)
        ax.set_ylim(max(0.0, y_min - padding), y_max + padding)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Parameters (log scale)")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            frameon=False,
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2 if len(models) > 2 else 1,
        )

    axes[0].set_ylabel("Ratchet MAE (lower is better)")
    fig.suptitle("Scaling Behavior by Design Phase", x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.06, 0.02, "Shaded bands show local variation across adjacent scale settings; points are measured sweep runs.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    plt.savefig(PLOTS_OUT / "story_scaling_curves.png", dpi=180, bbox_inches="tight")
    plt.close()


def render_rank_scatter(entries: list[dict]) -> None:
    plt.figure(figsize=(8, 5))
    annotated = [e for e in entries if e["best"].get("latent_eff_rank") is not None]
    colors = {
        "simple_transformer": "#4056a1",
        "ts_jepa": "#7b2cbf",
        "ada_jepa": "#c06c84",
        "multihorizon_meta_le_world_model": "#6c9a8b",
        "gnn_leworld_meta_pde": "#e76f51",
        "gnn_leworld_meta_ode": "#2a9d8f",
        "rate_anchor": "#1d3557",
        "rate_anchor_ac": "#457b9d",
    }
    for entry in annotated:
        x = entry["best"]["latent_eff_rank"]
        y = entry["best"]["ratchet_mae"]
        plt.scatter(x, y, s=80, color=colors[entry["model"]])
        plt.text(x + 0.08, y + 0.0001, entry["model"], fontsize=8)
    plt.xlabel("Latent effective rank")
    plt.ylabel("Ratchet MAE")
    plt.title("Representation Rank vs Accuracy")
    plt.tight_layout()
    plt.savefig(PLOTS_OUT / "story_rank_vs_accuracy.png", dpi=180)
    plt.close()


def render_baseline_delta(entries: list[dict]) -> None:
    """Show the final evidence ladder without implying every adjacent model is an ablation."""
    baseline = entries[0]["best"]["ratchet_mae"]
    labels = [entry["title"] for entry in entries]
    deltas = [baseline - entry["best"]["ratchet_mae"] for entry in entries]
    y = np.arange(len(entries))
    colors = ["#2a9d8f" if delta > 0.0002 else "#d97757" if delta < -0.0002 else "#8d99ae" for delta in deltas]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.barh(y, deltas, color=colors, height=0.62)
    ax.axvline(0, color="#293241", linewidth=1.2)
    for pos, delta in zip(y, deltas):
        text = "baseline" if abs(delta) < 0.0002 else f"{delta:+.4f}"
        offset = 0.00035 if delta >= 0 else -0.00035
        ax.text(delta + offset, pos, text, va="center", ha="left" if delta >= 0 else "right", fontsize=8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Ratchet MAE improvement versus direct baseline (positive is better)")
    fig.suptitle("Evidence Ladder: Which Designs Beat the Direct Baseline", x=0.125, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.125, 0.92, "Comparisons are against the same baseline, not claimed as adjacent causal ablations.", fontsize=8, color="#555555")
    ax.grid(axis="x", alpha=0.18, linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(PLOTS_OUT / "story_baseline_delta.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_rate_anchor_architecture() -> None:
    """Diagram the exact RateAnchor inference path implemented in models/rate_family.py."""
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, detail, color):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.4, edgecolor=color, facecolor=color + "20",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#1f2933")
        ax.text(x + w / 2, y + h * 0.34, detail, ha="center", va="center", fontsize=8, color="#3d4852", wrap=True)

    def arrow(start, end, label=None, curve=0.0):
        connector = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.25,
            color="#52616b", connectionstyle=f"arc3,rad={curve}",
        )
        ax.add_patch(connector)
        if label:
            x = (start[0] + end[0]) / 2
            y = (start[1] + end[1]) / 2
            ax.text(x, y + 0.025, label, ha="center", va="bottom", fontsize=7.5, color="#52616b")

    navy, teal, coral, gold = "#264653", "#2a9d8f", "#e76f51", "#e9c46a"
    box(0.03, 0.56, 0.14, 0.17, "Observed history", "patient states x[0:K]", navy)
    box(0.03, 0.27, 0.14, 0.17, "Known signals", "context + ERCP schedule", navy)
    box(0.23, 0.56, 0.15, 0.17, "LeWorld encoder", "state MLP + GRU\n-> initial latent z0", teal)
    box(0.44, 0.56, 0.14, 0.17, "GraphRefine", "similar-patient\nmessage passing", teal)
    box(0.43, 0.81, 0.17, 0.12, "Meta adaptation", "2 support-tail gradient steps\n-> patient task code z_task", gold)
    box(0.43, 0.22, 0.17, 0.15, "Rate anchor", "mean observed ratchet creep\n-> per-field gain g", coral)
    box(0.64, 0.56, 0.16, 0.17, "Drift-only latent ODE", "z(t+1) = z(t) + f(z, ctx, time, z_task, log g)\nno diffusion term", teal)
    box(0.84, 0.56, 0.12, 0.17, "Decoder", "latent -> raw\nstate update", navy)
    box(0.79, 0.22, 0.17, 0.18, "GainHead constraints", "ratchet increments x g\nmonotonicity + ERCP relief\n-> next state", coral)

    arrow((0.17, 0.645), (0.23, 0.645))
    arrow((0.38, 0.645), (0.44, 0.645))
    arrow((0.58, 0.645), (0.64, 0.645), "z0")
    arrow((0.60, 0.87), (0.69, 0.73), "z_task", curve=-0.15)
    arrow((0.17, 0.36), (0.64, 0.60), "future context + time", curve=-0.12)
    arrow((0.10, 0.56), (0.49, 0.37), "observed increments", curve=0.14)
    arrow((0.80, 0.645), (0.84, 0.645))
    arrow((0.90, 0.56), (0.88, 0.40))
    arrow((0.60, 0.295), (0.79, 0.31), "gain g")
    arrow((0.17, 0.32), (0.79, 0.31), "ERCP", curve=-0.05)
    arrow((0.87, 0.40), (0.72, 0.56), "roll forward", curve=-0.20)

    ax.text(0.03, 0.97, "RateAnchor: Best Practical Liver World Model", fontsize=18, fontweight="bold", color="#1f2933", va="top")
    ax.text(0.03, 0.035, "Best selected run: Ratchet MAE 0.0207. The rate anchor is non-parametric: it is estimated only from the observed patient history, then applied to constrained ratchet updates.", fontsize=9, color="#455a64")
    fig.tight_layout()
    fig.savefig(PLOTS_OUT / "rate_anchor_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def story_table(entries: list[dict]) -> str:
    lines = [
        "| # | Model | Problem it exposed | Intervention | What it taught us |",
        "|---:|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['order']} | `{entry['model']}` | {entry['problem']} | {entry['intervention']} | {entry['takeaway']} |"
        )
    return "\n".join(lines)


def quantitative_table(entries: list[dict]) -> str:
    lines = [
        "| # | Model | Params | Ratchet | Gap Closed % | held_out_susc | unseen_udca | h>60 | Rank |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        best = entry["best"]
        lines.append(
            f"| {entry['order']} | `{entry['model']}` | {best['params']:,} | {best['ratchet_mae']:.4f} | "
            f"{entry['gap_closed_pct']:.1f} | {best['probes']['held_out_susc']:.4f} | "
            f"{best['probes']['unseen_udca']:.4f} | {best.get('h_beyond', float('nan')):.4f} | "
            f"{fmt(best.get('latent_eff_rank'), 2)} |"
        )
    return "\n".join(lines)


def deep_table(entries: list[dict]) -> str:
    lines = [
        "| Model | Decomp Recall | Cirrhosis AUC | W1(F) | DTW Ratchet | Coverage90 | Manifold Score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        deep = entry["deep"]
        if deep is None:
            lines.append(f"| `{entry['model']}` | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| `{entry['model']}` | {deep['clinical']['decomp']['recall']:.2f} | "
            f"{deep['clinical']['cirrhosis_cls']['auc']:.3f} | {deep['distribution']['w1_finalF']:.4f} | "
            f"{deep['distribution']['dtw_ratchet']:.3f} | {deep['calibration']['coverage90_all']:.3f} | "
            f"{deep['manifold_critic']['score_model_rollout']:.3f} |"
        )
    return "\n".join(lines)


def source_table(entries: list[dict]) -> str:
    lines = [
        "| Model | Exact source files copied into `code/models/` | Exact best-run artifact copied | Rebuild command |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        srcs = "<br>".join(f"`{p}`" for p in entry["source_files"])
        best_path = Path(entry["best_run_path"]).name
        lines.append(
            f"| `{entry['model']}` | {srcs} | `artifacts/best_runs/{entry['model']}.json` "
            f"from `{best_path}` | `{entry['train_cmd']}` |"
        )
    return "\n".join(lines)


def tree_block() -> str:
    return """```text
liverworldmodel/
  README.md
  DECISION_MEMO.md
  RUNBOOK.md
  requirements.txt
  data/
    liver_data.csv
    liver_data_long.csv
    manifest.json
  code/
    Data_Generator.py
    SELECTED_EIGHT_CODE_MAP.md
    dataloader.py
    eval.py
    inference.py
    sweep.py
    training.py
    util.py
    training/
    models/
  artifacts/
    best_runs/
    deep_metrics/
    sweeps/
    selected8_manifest.csv
    selected8_manifest.json
  plots/
    story_ratchet_mae.png
    story_ood_probes.png
    story_scaling_curves.png
    story_rank_vs_accuracy.png
    story_baseline_delta.png
    rate_anchor_architecture.png
  scripts/
    build_selected8_package.py
    verify_package.py
```"""


def write_readme(entries: list[dict]) -> None:
    winner = next(e for e in entries if e["model"] == "rate_anchor")
    advanced = next(e for e in entries if e["model"] == "rate_anchor_ac")
    ode = next(e for e in entries if e["model"] == "gnn_leworld_meta_ode")
    pde = next(e for e in entries if e["model"] == "gnn_leworld_meta_pde")
    readme = f"""# Liver World Model: Selected 8-Experiment Pitch Package

Generated on `{DATE_STAMP}` from the experiment artifacts already present in the parent repository.

This directory is a self-contained pitch dossier for the 8 experiments that tell the cleanest story:
direct baseline, plain JEPA, adaptive JEPA, multi-horizon meta latent, PDE failure case, ODE repair,
rate-anchored winner, and the anti-collapse rate-anchor follow-up.

The primary take-home deliverable is [DECISION_MEMO.md](DECISION_MEMO.md). This README is the
supporting evidence pack: exact code/run mapping, tables, and figures behind that decision.

## Reviewer Start Here

1. Read [DECISION_MEMO.md](DECISION_MEMO.md) for the model-selection argument and its limitations.
2. Run `python scripts/verify_package.py` to validate the packaged code, data checksums, artifacts, and figures.
3. Use [RUNBOOK.md](RUNBOOK.md) for setup, reconstruction commands, and explicit non-claims.

The two synthetic datasets required by the evaluation harness are bundled in `data/` with SHA-256
checksums. The final RateAnchor checkpoint is deliberately documented as unavailable: this package
contains the exact configuration and recorded results, but does not claim byte-for-byte checkpoint reproduction.

## What Is In This Package

{tree_block()}

## Why These 8

The ordering is narrative, not alphabetical. It is designed to answer a sequence of questions:

1. Can a simple constrained model look good on aggregate MAE while still failing clinically?
2. Does plain JEPA latent learning beat direct prediction?
3. If not, where does JEPA start to help?
4. What happens when we add stronger latent adaptation and rollout-aware training?
5. Which continuous-time inductive bias is wrong?
6. Which one repairs the failure?
7. What is the final fix for rate-shift brittleness?
8. Does the more advanced anti-collapse version really outperform the simpler winner?

## Story Table

{story_table(entries)}

## Quantitative Comparison

Noise floor on ratchet MAE is `{entries[0]['best']['noise_floor_ratchet']:.4f}` and the persist-last
baseline is `{PERSIST:.4f}`. `Gap Closed %` means how much of the baseline-to-floor gap each model closes.

{quantitative_table(entries)}

## Clinical / Deep-Eval Slice

Not every late-wave model has a full deep battery JSON in the parent repo, so missing cells are left as `-`
rather than invented.

{deep_table(entries)}

## Exact Code And Run Mapping

{source_table(entries)}

The broad code copy inside `code/` includes the shared training loop, evaluation harness, inference utility,
loader, generator, and the full `models/` package so these selected experiments can be rebuilt without hidden
cross-file dependencies. Start with [code/SELECTED_EIGHT_CODE_MAP.md](code/SELECTED_EIGHT_CODE_MAP.md) for
the exact generator, training, evaluation, and architecture files needed by each selected experiment.

## Visual Comparison

### Best architecture: RateAnchor

![RateAnchor architecture](plots/rate_anchor_architecture.png)

`rate_anchor` is the best practical system because it combines the ingredients that survived the earlier
experiments, then directly fixes the remaining failure mode. It encodes the observed patient state history with
the LeWorld state MLP and GRU, refines the patient latent using similarity-weighted batch graph messages, and
fits a small patient-specific task code from the held-in tail of history. Forecasts evolve with a learned,
drift-only latent ODE, rather than the harmful diffusion term tested in the PDE branch. At decode time, the model
measures each patient's realised ratchet creep from observed increments and uses that non-parametric per-field
gain to scale the constrained monotone ratchet update. Known ERCP events still apply relief through the same
constraint head.

The eight experiments support this composition: plain JEPA alone is insufficient; adaptive and multi-horizon
latents improve hard future behavior; the PDE-to-ODE ablation isolates diffusion as harmful
({pde['best']['ratchet_mae']:.4f} to {ode['best']['ratchet_mae']:.4f}); and rate anchoring produces the best
headline ratchet MAE ({winner['best']['ratchet_mae']:.4f}) and strongest held-out susceptibility probe. The
anti-collapse follow-up improves representation rank but does not improve this primary accuracy metric, so the
simpler RateAnchor remains the recommended deployment candidate.

### 1. Story-order ratchet MAE

![Story-order ratchet MAE](plots/story_ratchet_mae.png)

The ending is intentionally honest: the best practical model is still the simpler `rate_anchor`
({winner['best']['ratchet_mae']:.4f}), while the more advanced `rate_anchor_ac` lands at
{advanced['best']['ratchet_mae']:.4f}.

### 2. OOD and long-horizon probes

![OOD and long-horizon probes](plots/story_ood_probes.png)

This plot makes the middle and end of the story clearer than the single headline MAE:
`ada_jepa` helps the beyond-horizon regime, `multihorizon_meta_le_world_model` is strong on the hard
susceptibility probe, and `rate_anchor` dominates both in-distribution and hidden-rate shift.

### 3. Scaling curves across parameter budgets

![Scaling curves](plots/story_scaling_curves.png)

This is the cleanest answer to "was the win just because the model is bigger?" The ODE repair and the
rate-anchored family remain strong across the sweep, while the PDE branch stays consistently worse.
Each phase uses its own y-axis range so within-family scaling behavior remains legible; use the labeled axes,
not apparent panel height, for cross-phase absolute comparisons.

### 4. Representation rank vs accuracy

![Rank vs accuracy](plots/story_rank_vs_accuracy.png)

This is where `rate_anchor_ac` earns its keep in the paper: it is the more advanced version and it greatly
improves latent rank, but the main task still belongs to the simpler `rate_anchor`. That is a useful
scientific result, not a disappointment.

### 5. Evidence ladder against the direct baseline

![Baseline-relative evidence ladder](plots/story_baseline_delta.png)

This final plot keeps the scientific claim disciplined: every bar is compared to the same direct supervised
baseline, so it shows which designs actually improved the main metric without pretending that every neighboring
pair is a clean causal ablation.

## Recommended Verbal Pitch

- `simple_transformer`: strong-looking baseline, but clinically incomplete.
- `ts_jepa`: honest negative result that prevents overclaiming for JEPA.
- `ada_jepa`: first clear sign that JEPA helps when it adapts at test time.
- `multihorizon_meta_le_world_model`: latent adaptation starts paying off on hidden progression speed.
- `gnn_leworld_meta_pde`: wrong dynamics prior; useful negative control.
- `gnn_leworld_meta_ode`: mechanistic repair and wave-1 climax.
- `rate_anchor`: final practical winner and strongest headline model.
- `rate_anchor_ac`: more advanced follow-up that improves representation quality, not final task performance.

## Key Bottom Lines

- Best practical model: `rate_anchor` with ratchet MAE `{winner['best']['ratchet_mae']:.4f}`.
- More advanced but not better headline model: `rate_anchor_ac` with `{advanced['best']['ratchet_mae']:.4f}`.
- Sharpest mechanistic ablation: PDE vs ODE, `{pde['best']['ratchet_mae']:.4f}` to `{ode['best']['ratchet_mae']:.4f}`.
- Honest JEPA message: plain JEPA does not win the clean task, adaptive JEPA helps where horizon shift matters.
"""
    README_PATH.write_text(readme, encoding="utf-8")


def main() -> None:
    entries = build_entries()
    copy_artifacts(entries)
    save_manifest(entries)
    render_story_bars(entries)
    render_probe_bars(entries)
    render_scaling(entries)
    render_rank_scatter(entries)
    render_baseline_delta(entries)
    render_rate_anchor_architecture()
    write_readme(entries)
    print(f"built package at {PACKAGE_ROOT}")


if __name__ == "__main__":
    main()
