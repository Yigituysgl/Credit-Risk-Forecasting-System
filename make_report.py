"""Build the evaluation charts in docs/ from the saved models.

Usage (after train.py):
    python make_report.py --data path/to/accepted_2007_to_2018Q4.csv

Recreates the same fit / calibration / test split as train.py, scores the
test period with the models saved under models/, checks the scores
reproduce reports/metrics.json, and writes four PNGs:

    model_comparison.png    ROC and precision-recall curves
    calibration.png         reliability curves before/after calibration
    default_rate_by_decile.png   observed vs predicted default rate by risk decile
    sensitivity.png         ROC AUC and KS with and without LC pricing features
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from credit_risk.dataset import load_splits  # noqa: E402
from credit_risk.modeling import evaluate  # noqa: E402
from credit_risk.target import TARGET  # noqa: E402

MODELS = {"xgboost": "XGBoost", "logistic_regression": "Logistic regression"}
FEATURE_SETS = {"full": "All features", "without_pricing": "Without LC pricing"}

# Reference palette (dataviz skill), light mode: categorical slots 1-2, chrome and ink.
COLORS = {"xgboost": "#2a78d6", "logistic_regression": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})
DOT = {"markersize": 8, "markeredgecolor": SURFACE, "markeredgewidth": 1.5}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="path to accepted_2007_to_2018Q4.csv")
    p.add_argument("--models-dir", default=Path("models"), type=Path)
    p.add_argument("--metrics", default=Path("reports/metrics.json"), type=Path)
    p.add_argument("--out-dir", default=Path("docs"), type=Path)
    return p.parse_args(argv)


def _pct(x, _=None):
    return f"{x:.0%}"


def _title(fig, title, subtitle):
    fig.suptitle(title, x=0.01, y=0.99, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=10, color=INK_2)


def _save(fig, path):
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)
    print(f"Wrote {path}")


def score_test(splits, models_dir):
    """Test-period scores for every feature set, model and calibration stage."""
    test = splits["test"]
    scores = {}
    for set_name in FEATURE_SETS:
        for name in MODELS:
            for stage, suffix in (("uncalibrated", ""), ("calibrated", "_calibrated")):
                model = joblib.load(models_dir / set_name / f"{name}{suffix}.joblib")
                cols = list(model.feature_names_in_)
                scores[set_name, name, stage] = model.predict_proba(test[cols])[:, 1]
    return scores


def check_against_metrics(scores, y, metrics):
    """Fail if the saved models do not reproduce the numbers in metrics.json."""
    for (set_name, name, stage), p in scores.items():
        got = evaluate(y, p)
        want = metrics["models"][set_name][name][stage]
        diff = {k: (got[k], want[k]) for k in want if abs(got[k] - want[k]) > 1e-4}
        if diff:
            raise ValueError(f"{set_name}/{name}/{stage} does not match metrics.json: {diff}")


def plot_model_comparison(scores, y, metrics, path):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.07, right=0.98, wspace=0.25)
    for name, label in MODELS.items():
        p = scores["full", name, "calibrated"]
        m = metrics["models"]["full"][name]["calibrated"]
        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, color=COLORS[name], label=f"{label}  (AUC {m['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax_pr.plot(rec, prec, color=COLORS[name], label=f"{label}  (PR AUC {m['pr_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax_roc.text(0.62, 0.55, "Random", color=MUTED, rotation=33, fontsize=9)
    base = float(np.mean(y))
    ax_pr.axhline(base, color=MUTED, linewidth=1)
    ax_pr.text(0.5, base - 0.045, f"Base rate {base:.1%}", color=MUTED, ha="center", fontsize=9)
    ax_roc.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    ax_pr.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    for ax in (ax_roc, ax_pr):
        ax.legend(loc="lower right" if ax is ax_roc else "upper right")
    s = metrics["split"]
    _title(fig, "Model comparison on the 2015 test set",
           f"Out-of-time evaluation on {len(y):,} loans issued {s['test']['issue_d_min']} to {s['test']['issue_d_max']}; "
           f"fit on {s['fit']['issue_d_min']} to {s['fit']['issue_d_max']}, "
           f"calibrated on {s['calibration']['issue_d_min']} to {s['calibration']['issue_d_max']}.")
    _save(fig, path)


def _reliability(y, p, bins=10):
    """Mean predicted vs observed default rate in equal-count bins of predicted PD."""
    df = pd.DataFrame({"y": np.asarray(y), "p": p})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"))
    return g["pred"].to_numpy(), g["obs"].to_numpy()


def plot_calibration(scores, y, metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.07, right=0.98, wspace=0.08)
    lim = 0.5
    for ax, (name, label) in zip(axes, MODELS.items()):
        ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1)
        ax.text(lim * 0.97, lim * 0.90, "Perfect calibration", color=MUTED, ha="right", rotation=38, fontsize=9)
        for stage, color in (("uncalibrated", MUTED), ("calibrated", COLORS[name])):
            m = metrics["models"]["full"][name][stage]
            pred, obs = _reliability(y, scores["full", name, stage])
            # Uncalibrated: hollow gray rings; calibrated: filled model colour with a surface ring.
            style = (
                {"markerfacecolor": SURFACE, "markeredgecolor": MUTED, "markeredgewidth": 2, "markersize": 7}
                if stage == "uncalibrated"
                else {**DOT, "markerfacecolor": color}
            )
            ax.plot(pred, obs, color=color, marker="o", **style,
                    label=f"{stage.capitalize()}: mean PD {m['mean_predicted_pd']:.1%}, Brier {m['brier']:.4f}")
        ax.set(title=label, xlabel="Mean predicted default rate (decile)", xlim=(0, lim), ylim=(0, lim))
        ax.xaxis.set_major_formatter(_pct)
        ax.yaxis.set_major_formatter(_pct)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Observed default rate")
    _title(fig, "Calibration before and after Platt scaling",
           f"Calibrated on {metrics['split']['calibration']['issue_d_min']} to "
           f"{metrics['split']['calibration']['issue_d_max']} loans; evaluated on the test set "
           f"(observed default rate {np.mean(y):.1%}). "
           "Points above the diagonal = under-prediction.")
    _save(fig, path)


def plot_deciles(scores, y, path):
    p = scores["full", "xgboost", "calibrated"]
    df = pd.DataFrame({"y": np.asarray(y), "p": p})
    df["decile"] = pd.qcut(df["p"], 10, labels=False) + 1
    g = df.groupby("decile").agg(obs=("y", "mean"), pred=("p", "mean"), n=("y", "size"))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.07, right=0.98)
    x = g.index.to_numpy()
    ax.bar(x, g["obs"], width=0.6, color=COLORS["xgboost"], label="Observed default rate", zorder=2)
    ax.plot(x, g["pred"], color=INK_2, linewidth=0, marker="o", markerfacecolor=INK_2, **DOT,
            label="Mean predicted PD (calibrated)", zorder=3)
    for xi in (x[0], x[-1]):
        ax.annotate(f"{g.loc[xi, 'obs']:.1%}", (xi, g.loc[xi, "obs"]), textcoords="offset points",
                    xytext=(0, 6), ha="center", color=INK, fontsize=9, fontweight="semibold")
    ax.set(xticks=x, xlabel="Risk decile (1 = lowest predicted risk)", ylabel="Default rate")
    ax.yaxis.set_major_formatter(_pct)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ratio = g["obs"].iloc[-1] / g["obs"].iloc[0]
    _title(fig, "Default rate by XGBoost risk decile, 2015 test set",
           f"Each decile holds ~{int(g['n'].median()):,} loans. "
           f"The riskiest decile defaults {ratio:.1f}x as often as the safest.")
    _save(fig, path)


def plot_sensitivity(metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.subplots_adjust(top=0.70, bottom=0.17, left=0.17, right=0.97, wspace=0.12)
    rows = list(MODELS)
    for ax, (key, label) in zip(axes, (("roc_auc", "ROC AUC"), ("ks", "KS statistic"))):
        for i, name in enumerate(rows):
            full = metrics["models"]["full"][name]["calibrated"][key]
            red = metrics["models"]["without_pricing"][name]["calibrated"][key]
            ax.plot([red, full], [i, i], color=GRID, linewidth=4, solid_capstyle="round", zorder=1)
            ax.plot(full, i, "o", color=COLORS[name], markerfacecolor=COLORS[name], **DOT, zorder=3)
            ax.plot(red, i, "o", color=COLORS[name], markerfacecolor="white", markeredgewidth=2,
                    markeredgecolor=COLORS[name], markersize=8, zorder=3)
            ax.annotate(f"{full:.3f}", (full, i), textcoords="offset points", xytext=(8, -3), color=INK, fontsize=9)
            ax.annotate(f"{red:.3f}", (red, i), textcoords="offset points", xytext=(-8, -3), ha="right",
                        color=INK_2, fontsize=9)
        lo = min(metrics["models"]["without_pricing"][n]["calibrated"][key] for n in rows)
        hi = max(metrics["models"]["full"][n]["calibrated"][key] for n in rows)
        ax.set_xlim(lo - 0.02, hi + 0.02)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([MODELS[n] for n in rows] if ax is axes[0] else [])
        ax.set_title(label)
        ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], linestyle="", marker="o", color=INK_2, markersize=8, label="All features"),
        plt.Line2D([], [], linestyle="", marker="o", color=INK_2, markerfacecolor="white", markeredgewidth=2,
                   markersize=8, label="Without int_rate, installment, sub_grade"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.86), ncol=2)
    _title(fig, "What the model adds beyond Lending Club's own pricing", "2015 test set, calibrated models.")
    _save(fig, path)


def main(argv=None):
    args = parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    print(f"Loading {args.data.name} ...")
    splits, _ = load_splits(
        args.data,
        pd.Timestamp(metrics["data"]["window_start"]),
        pd.Timestamp(metrics["data"]["calibration_start"]),
        pd.Timestamp(metrics["data"]["test_start"]),
    )
    y = splits["test"][TARGET]
    scores = score_test(splits, args.models_dir)
    check_against_metrics(scores, y, metrics)
    print("Saved models reproduce metrics.json")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(scores, y, metrics, args.out_dir / "model_comparison.png")
    plot_calibration(scores, y, metrics, args.out_dir / "calibration.png")
    plot_deciles(scores, y, args.out_dir / "default_rate_by_decile.png")
    plot_sensitivity(metrics, args.out_dir / "sensitivity.png")


if __name__ == "__main__":
    main()
