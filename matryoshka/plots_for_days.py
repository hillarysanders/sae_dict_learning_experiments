#!/usr/bin/env python3
"""
make_plots.py

Generate "headline" plots for the SAE experiments in this repo.

It expects you've run the README commands, which produce:
- Toy results:
    runs/toy_absorption_<run_name>.json
- Train results:
    runs/<run_name>_<timestamp>/
        metrics.jsonl
        eval.json
        sae_step_<step>.pt

Plots are saved to: <out_dir>/ (default: runs/plots)

Key plots:
1) Toy Pareto scatter: avg_max_decoder_cos vs fvu
2) Toy absorption vs L0: absorption_rate_latent_contrib vs l0_mean
3) Train dashboards over checkpoints: fvu vs step, l0_mean vs step, avg_max_decoder_cos vs step
4) Train trajectory: avg_max_decoder_cos vs fvu over checkpoints (line)
5) Schedules: p_current vs step; lambda_mean/min/max vs step (from metrics.jsonl)

Notes on "significance":
- If you have multiple repeat runs per run_name (multiple timestamped dirs),
  the script aggregates end-of-run eval metrics and plots mean ± std error bars.
- For training dynamics, it currently picks the *most recent* run dir per run_name
  unless you set --use_all_train_runs (then it will evaluate checkpoints for each run dir).

This script uses matplotlib only (no seaborn), and does not hardcode colors.
"""

from __future__ import annotations
import torch
from dataclasses import fields
from typing import Any, Dict
from config import Config
import argparse
import json
import math
import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

# Repo imports (these must exist in your project)
from config import Config
from sae import SparseAutoencoder
from tl_activations import pick_device, pick_dtype, load_tl_model
from eval_metrics import run_quick_eval


# ---------------------------
# Helpers: IO + parsing
# ---------------------------

def _safe_read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _list_run_dirs(runs_dir: Path, run_name: str) -> List[Path]:
    # make_run_dir: runs/<run_name>_<stamp>
    # also accept exact matches if user created custom folders
    out = []
    pat = re.compile(rf"^{re.escape(run_name)}(_\d{{8}}_\d{{6}})?$")
    for p in runs_dir.iterdir():
        if p.is_dir() and pat.match(p.name):
            out.append(p)
        elif p.is_dir() and p.name.startswith(run_name + "_"):
            out.append(p)
    return sorted(out)


def _most_recent_run_dir(run_dirs: List[Path]) -> Optional[Path]:
    if not run_dirs:
        return None
    # Timestamp is in name; sorting lexicographically works for YYYYMMDD_HHMMSS
    return sorted(run_dirs)[-1]


def _read_metrics_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def _cfg_from_dict(cfg_dict: Dict[str, Any]) -> Config:
    """
    Robustly reconstruct Config from a dict saved in checkpoints.

    Important: dataclass fields with init=False (like run_dir) are NOT accepted by Config(...),
    even though they appear in fields(Config). So we must only pass init=True fields.
    """
    init_field_names = {f.name for f in fields(Config) if f.init}
    kwargs = {k: v for k, v in cfg_dict.items() if k in init_field_names}

    cfg = Config(**kwargs)  # type: ignore[arg-type]

    # Best-effort: restore non-init fields after construction (harmless if absent)
    for f in fields(Config):
        if not f.init and f.name in cfg_dict:
            try:
                setattr(cfg, f.name, cfg_dict[f.name])
            except Exception:
                pass

    return cfg


def _checkpoint_step(path: Path) -> int:
    # sae_step_500.pt -> 500
    m = re.search(r"sae_step_(\d+)\.pt$", path.name)
    if not m:
        return -1
    return int(m.group(1))


def _list_checkpoints(run_dir: Path) -> List[Path]:
    ckpts = sorted(run_dir.glob("sae_step_*.pt"), key=_checkpoint_step)
    return [p for p in ckpts if _checkpoint_step(p) >= 0]


# ---------------------------
# Plot helpers
# ---------------------------

def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _annotate_points(ax, xs, ys, labels, *, max_labels: int = 40) -> None:
    # Avoid unreadable label soup if too many points
    if len(labels) > max_labels:
        return
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(str(lab), (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(max(v, 0.0))


# ---------------------------
# Toy plots
# ---------------------------

def load_all_toy_results(runs_dir: Path) -> List[Dict[str, Any]]:
    toy_files = sorted(runs_dir.glob("toy_absorption_*.json"))
    rows: List[Dict[str, Any]] = []

    for tf in toy_files:
        try:
            payload = _safe_read_json(tf)  # Dict[tag -> metrics dict]
        except Exception:
            continue
        for tag, metrics in payload.items():
            row = {"toy_file": tf.name, "tag": tag}
            row.update(metrics)
            rows.append(row)
    return rows

def plot_toy_pareto(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Plot toy Pareto scatter:
      - faint individual points (colored by tag)
      - bold mean point per tag (same color)
      - optional error bars (std) per tag (x=fvu, y=avg_max_decoder_cos)
      - annotate ONLY the mean points (one label per tag)
    """
    # tag -> list of (x, y)
    by_tag: Dict[str, List[Tuple[float, float]]] = {}

    for r in rows:
        if "fvu" in r and "avg_max_decoder_cos" in r:
            tag = str(r.get("tag", "?"))
            x = float(r["fvu"])
            y = float(r["avg_max_decoder_cos"])
            by_tag.setdefault(tag, []).append((x, y))

    if not by_tag:
        print("[toy] No rows with fvu and avg_max_decoder_cos found; skipping toy pareto plot.")
        return

    plt.figure()
    ax = plt.gca()

    mean_xs, mean_ys, mean_labels = [], [], []

    # Matplotlib will cycle colors once per mean scatter call, so we ensure:
    #  - individual points for a tag use the *same* color as that tag's mean point
    for tag in sorted(by_tag.keys()):
        pts = by_tag[tag]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # First: plot mean point (captures the next color in the cycle)
        mx, sx = _mean_std(xs)
        my, sy = _mean_std(ys)

        # Mean point + error bars (std). Error bars are harmless if std==0.
        eb = ax.errorbar(
            [mx],
            [my],
            xerr=[sx] if sx > 0 else None,
            yerr=[sy] if sy > 0 else None,
            fmt="o",
            markersize=7,
            capsize=3,
            label=f"{tag} (mean ± std)",
        )
        # Get the color used for this tag so we can reuse it for faint points
        color = eb[0].get_color()

        # Then: plot individual points (same color, transparent, no labels)
        ax.scatter(xs, ys, alpha=0.18, s=18, color=color)

        mean_xs.append(mx)
        mean_ys.append(my)
        mean_labels.append(tag)

    ax.set_xlabel("FVU (lower better)")
    ax.set_ylabel("AvgMax decoder cosine (lower better)")
    ax.set_title("Toy: Redundancy vs Reconstruction (means + per-run scatter)")
    _annotate_points(ax, mean_xs, mean_ys, mean_labels, max_labels=60)
    ax.legend(loc="best", fontsize=8)
    _savefig(out_dir / "toy_pareto_redundancy_vs_fvu.png")


def plot_toy_absorption_vs_l0(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Plot toy absorption vs sparsity:
      - faint individual points (colored by tag)
      - bold mean point per tag (same color)
      - optional error bars (std) per tag (x=l0_mean, y=absorption_rate_latent_contrib)
      - annotate ONLY the mean points (one label per tag)
    """
    by_tag: Dict[str, List[Tuple[float, float]]] = {}

    for r in rows:
        if "l0_mean" in r and "absorption_rate_latent_contrib" in r:
            tag = str(r.get("tag", "?"))
            x = float(r["l0_mean"])
            y = float(r["absorption_rate_latent_contrib"])
            by_tag.setdefault(tag, []).append((x, y))

    if not by_tag:
        print("[toy] No rows with l0_mean and absorption_rate_latent_contrib found; skipping absorption plot.")
        return

    plt.figure()
    ax = plt.gca()

    mean_xs, mean_ys, mean_labels = [], [], []

    for tag in sorted(by_tag.keys()):
        pts = by_tag[tag]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        mx, sx = _mean_std(xs)
        my, sy = _mean_std(ys)

        eb = ax.errorbar(
            [mx],
            [my],
            xerr=[sx] if sx > 0 else None,
            yerr=[sy] if sy > 0 else None,
            fmt="o",
            markersize=7,
            capsize=3,
            label=f"{tag} (mean ± std)",
        )
        color = eb[0].get_color()

        ax.scatter(xs, ys, alpha=0.18, s=18, color=color)

        mean_xs.append(mx)
        mean_ys.append(my)
        mean_labels.append(tag)

    ax.set_xlabel("L0 mean (avg active latents)")
    ax.set_ylabel("Absorption rate (latent contrib match) (lower better)")
    ax.set_title("Toy: Absorption vs Sparsity (means + per-run scatter)")
    _annotate_points(ax, mean_xs, mean_ys, mean_labels, max_labels=60)
    ax.legend(loc="best", fontsize=8)
    _savefig(out_dir / "toy_absorption_vs_l0.png")


# ---------------------------
# Train: end-of-run aggregation (significance-ish)
# ---------------------------

def collect_eval_summaries(
    runs_dir: Path, run_names: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rn in run_names:
        dirs = _list_run_dirs(runs_dir, rn)
        evals: List[Dict[str, Any]] = []
        for d in dirs:
            p = d / "eval.json"
            if p.exists():
                try:
                    ev = _safe_read_json(p)
                    ev["_run_dir"] = str(d)
                    evals.append(ev)
                except Exception:
                    continue
        out[rn] = evals
    return out


def print_eval_table(eval_by_run: Dict[str, List[Dict[str, Any]]], metrics: List[str]) -> None:
    print("\n=== End-of-run eval summary (mean ± std over repeats) ===")
    header = ["run_name", "n"] + metrics
    print("\t".join(header))
    for rn, items in eval_by_run.items():
        n = len(items)
        row = [rn, str(n)]
        for m in metrics:
            vals = []
            for it in items:
                if m in it:
                    try:
                        vals.append(float(it[m]))
                    except Exception:
                        pass
            mu, sd = _mean_std(vals)
            if math.isnan(mu):
                row.append("NA")
            else:
                row.append(f"{mu:.4f} ± {sd:.4f}" if n > 1 else f"{mu:.4f}")
        print("\t".join(row))


def plot_end_eval_bars(
    eval_by_run: Dict[str, List[Dict[str, Any]]],
    metric: str,
    out_dir: Path,
    *,
    title: Optional[str] = None,
) -> None:
    names = []
    means = []
    stds = []

    for rn, items in eval_by_run.items():
        vals = []
        for it in items:
            if metric in it:
                try:
                    vals.append(float(it[metric]))
                except Exception:
                    pass
        if not vals:
            continue
        mu, sd = _mean_std(vals)
        names.append(rn)
        means.append(mu)
        stds.append(sd)

    if not names:
        return

    plt.figure(figsize=(max(6, 0.8 * len(names)), 4))
    ax = plt.gca()
    x = list(range(len(names)))
    ax.bar(x, means, yerr=stds if any(s > 0 for s in stds) else None, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"End-of-run: {metric} (mean ± std)")
    _savefig(out_dir / f"end_eval_{metric}.png")


# ---------------------------
# Train: checkpoint evaluation for dashboards + trajectory
# ---------------------------

def eval_checkpoints_for_run_dir(
    run_dir: Path,
    *,
    device_override: Optional[str],
    dtype_override: Optional[str],
    eval_num_batches_override: Optional[int],
) -> Tuple[List[int], List[Dict[str, float]]]:
    """
    Evaluate run_quick_eval at each checkpoint in run_dir.

    Returns:
        steps: list[int]
        evals: list[dict]
    """
    ckpts = _list_checkpoints(run_dir)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {run_dir}")

    # Load one checkpoint to get cfg + d_model
    ck0 = torch_load(ckpts[0])
    cfg_dict = ck0.get("cfg", {})
    cfg = _cfg_from_dict(cfg_dict)

    # Allow user override for plotting-time device/dtype
    device = pick_device(device_override or cfg.device)
    dtype = pick_dtype(dtype_override or cfg.dtype, device)

    # Override eval batch count to keep plots fast if desired
    if eval_num_batches_override is not None:
        cfg.eval_num_batches = int(eval_num_batches_override)

    model = load_tl_model(cfg.model_name, device=device, dtype=dtype)
    hook_name = cfg.resolved_hook()

    steps: List[int] = []
    evals: List[Dict[str, float]] = []

    for ck in ckpts:
        step = _checkpoint_step(ck)
        data = torch_load(ck)
        d_model = int(data["d_model"])
        state_dict = data["state_dict"]

        sae = SparseAutoencoder(
            d_model=d_model,
            n_latents=int(cfg.n_latents),
            tied_weights=bool(cfg.tied_weights),
            activation=cfg.activation,
            device=device,
            dtype=torch.float32,  # SAE params in fp32 for stability
        )
        sae.load_state_dict(state_dict)
        sae.eval()

        out = run_quick_eval(cfg, sae, model=model, hook_name=hook_name, theta=None)
        steps.append(step)
        evals.append({k: float(v) for k, v in out.items() if isinstance(v, (int, float))})

        print(f"[ckpt eval] {run_dir.name} step={step} -> "
              f"fvu={out.get('fvu', float('nan')):.4f} "
              f"l0={out.get('l0_mean', float('nan')):.2f} "
              f"cos={out.get('avg_max_decoder_cos', float('nan')):.4f}")

    return steps, evals


def plot_train_dashboard_over_checkpoints(
    run_label_to_series: Dict[str, Tuple[List[int], List[Dict[str, float]]]],
    out_dir: Path,
) -> None:
    """
    Creates three separate plots:
      - fvu vs step
      - l0_mean vs step
      - avg_max_decoder_cos vs step
    """
    for metric, ylabel, fname, title in [
        ("fvu", "FVU (lower better)", "train_dashboard_fvu_vs_step.png", "Train (checkpoints): FVU vs step"),
        ("l0_mean", "L0 mean (avg actives per token)", "train_dashboard_l0_vs_step.png", "Train (checkpoints): L0 vs step"),
        ("avg_max_decoder_cos", "AvgMax decoder cosine (lower better)", "train_dashboard_cos_vs_step.png", "Train (checkpoints): Decoder redundancy vs step"),
    ]:
        plt.figure()
        ax = plt.gca()
        any_plotted = False

        for label, (steps, evals) in run_label_to_series.items():
            ys = [float(e.get(metric, float("nan"))) for e in evals]
            if all(math.isnan(y) for y in ys):
                continue
            ax.plot(steps, ys, marker="o", label=label)
            any_plotted = True

        if not any_plotted:
            plt.close()
            continue

        ax.set_xlabel("Training step (checkpoint)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        _savefig(out_dir / fname)


def plot_train_trajectory(
    run_label_to_series: Dict[str, Tuple[List[int], List[Dict[str, float]]]],
    out_dir: Path,
) -> None:
    """
    Plot redundancy vs recon over checkpoints, with lines in checkpoint order.
      x = fvu
      y = avg_max_decoder_cos
    """
    plt.figure()
    ax = plt.gca()
    any_plotted = False

    for label, (_steps, evals) in run_label_to_series.items():
        xs = [float(e.get("fvu", float("nan"))) for e in evals]
        ys = [float(e.get("avg_max_decoder_cos", float("nan"))) for e in evals]
        if all(math.isnan(x) for x in xs) or all(math.isnan(y) for y in ys):
            continue
        ax.plot(xs, ys, marker="o", label=label)
        any_plotted = True

    if not any_plotted:
        plt.close()
        return

    ax.set_xlabel("FVU (lower better)")
    ax.set_ylabel("AvgMax decoder cosine (lower better)")
    ax.set_title("Train (checkpoints): Redundancy vs Reconstruction trajectory")
    ax.legend()
    _savefig(out_dir / "train_trajectory_redundancy_vs_fvu.png")


# ---------------------------
# Schedules from metrics.jsonl
# ---------------------------

def plot_schedules_from_metrics(
    run_label_to_metrics_rows: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
) -> None:
    # p_current vs step
    plt.figure()
    ax = plt.gca()
    any_plotted = False
    for label, rows in run_label_to_metrics_rows.items():
        xs, ys = [], []
        for r in rows:
            if "step" in r and ("p_current" in r):
                xs.append(int(r["step"]))
                ys.append(float(r["p_current"]))
        if xs:
            ax.plot(xs, ys, label=label)
            any_plotted = True
    if any_plotted:
        ax.set_xlabel("Step")
        ax.set_ylabel("p_current")
        ax.set_title("Schedules: p_current vs step")
        ax.legend()
        _savefig(out_dir / "schedule_p_current_vs_step.png")
    else:
        plt.close()

    # lambda stats vs step (mean/min/max when present)
    for key, ylabel, fname in [
        ("lambda_mean", "lambda_mean", "schedule_lambda_mean_vs_step.png"),
        ("lambda_min", "lambda_min", "schedule_lambda_min_vs_step.png"),
        ("lambda_max", "lambda_max", "schedule_lambda_max_vs_step.png"),
    ]:
        plt.figure()
        ax = plt.gca()
        any_plotted = False
        for label, rows in run_label_to_metrics_rows.items():
            xs, ys = [], []
            for r in rows:
                if "step" in r and key in r:
                    xs.append(int(r["step"]))
                    ys.append(float(r[key]))
            if xs:
                ax.plot(xs, ys, label=label)
                any_plotted = True
        if any_plotted:
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Schedules: {key} vs step")
            ax.legend()
            _savefig(out_dir / fname)
        else:
            plt.close()


# ---------------------------
# Torch load wrapper (keeps imports tidy)
# ---------------------------

def torch_load(path: Path) -> Dict[str, Any]:
    import torch
    return torch.load(path, map_location="cpu")


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs", help="Directory containing run outputs")
    ap.add_argument("--out_dir", type=str, default="runs/plots", help="Where to write plots")

    ap.add_argument(
        "--train_run_names",
        type=str,
        nargs="*",
        default=["mac_l1_uniform", "mac_p_anneal", "mac_freq_l1", "mac_combined"],
        help="Train run_name prefixes to include (folders are runs/<run_name>_<timestamp>/...)",
    )
    ap.add_argument(
        "--use_all_train_runs",
        action="store_true",
        help="If set, evaluate checkpoints for ALL timestamped dirs per run_name (slower). "
             "Otherwise uses most recent run dir per run_name for time-series plots.",
    )

    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device for checkpoint evaluation (e.g. cuda/cpu/mps). "
             "If not set, uses the device stored in each run's config.",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Override dtype for model loading during checkpoint eval (e.g. bf16/fp16/fp32). "
             "If not set, uses the dtype stored in each run's config.",
    )
    ap.add_argument(
        "--eval_num_batches",
        type=int,
        default=5,
        help="Override cfg.eval_num_batches for checkpoint eval (smaller=faster, noisier).",
    )

    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    # ----------------
    # Toy plots
    # ----------------
    toy_rows = load_all_toy_results(runs_dir)
    if toy_rows:
        plot_toy_pareto(toy_rows, out_dir)
        plot_toy_absorption_vs_l0(toy_rows, out_dir)
        print(f"[ok] Wrote toy plots to: {out_dir}")
    else:
        print("[toy] No toy_absorption_*.json found; skipping toy plots.")

    # ----------------
    # Train end-of-run summary (+ error bars if repeats)
    # ----------------
    eval_by_run = collect_eval_summaries(runs_dir, args.train_run_names)
    summary_metrics = [
        "fvu",
        "l0_mean",
        "avg_max_decoder_cos",
        "dead_frac_p_lt_1e_6",
        "rare_frac_p_lt_1e_4",
    ]
    print_eval_table(eval_by_run, summary_metrics)

    # End-of-run bar plots (mean ± std)
    for m in ["fvu", "l0_mean", "avg_max_decoder_cos", "dead_frac_p_lt_1e_6", "rare_frac_p_lt_1e_4"]:
        plot_end_eval_bars(eval_by_run, m, out_dir)

    # ----------------
    # Train checkpoint dashboards + trajectory (requires checkpoints)
    # ----------------
    run_label_to_series: Dict[str, Tuple[List[int], List[Dict[str, float]]]] = {}
    run_label_to_metrics_rows: Dict[str, List[Dict[str, Any]]] = {}

    for rn in args.train_run_names:
        dirs = _list_run_dirs(runs_dir, rn)
        if not dirs:
            print(f"[train] No run dirs found for {rn!r}")
            continue

        chosen_dirs = dirs if args.use_all_train_runs else [d for d in [ _most_recent_run_dir(dirs) ] if d is not None]
        for d in chosen_dirs:
            label = rn if (not args.use_all_train_runs) else f"{rn}:{Path(d).name}"

            # metrics.jsonl (for schedules)
            mpath = Path(d) / "metrics.jsonl"
            if mpath.exists():
                try:
                    run_label_to_metrics_rows[label] = _read_metrics_jsonl(mpath)
                except Exception as e:
                    print(f"[warn] failed reading metrics.jsonl for {d}: {e}")

            # checkpoint eval series
            ckpts = _list_checkpoints(Path(d))
            if not ckpts:
                print(f"[train] No checkpoints in {d} (expected sae_step_*.pt); skipping dashboard for this run.")
                continue

            try:
                steps, evals = eval_checkpoints_for_run_dir(
                    Path(d),
                    device_override=args.device,
                    dtype_override=args.dtype,
                    eval_num_batches_override=args.eval_num_batches,
                )
                run_label_to_series[label] = (steps, evals)
            except Exception as e:
                print(f"[warn] checkpoint eval failed for {d}: {e}")

    if run_label_to_series:
        plot_train_dashboard_over_checkpoints(run_label_to_series, out_dir)
        plot_train_trajectory(run_label_to_series, out_dir)
        print(f"[ok] Wrote checkpoint dashboard + trajectory plots to: {out_dir}")
    else:
        print("[train] No checkpoint series evaluated; skipping checkpoint dashboards/trajectory.")

    # ----------------
    # Schedules
    # ----------------
    if run_label_to_metrics_rows:
        plot_schedules_from_metrics(run_label_to_metrics_rows, out_dir)
        print(f"[ok] Wrote schedule plots to: {out_dir}")
    else:
        print("[train] No metrics.jsonl loaded; skipping schedule plots.")

    print("\nDone.")
    print(f"Plots directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
