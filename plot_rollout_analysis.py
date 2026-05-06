import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Serif",
        }
    )


def load_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "episode",
        "step",
        "mix_sim",
        "stealthiness_cost",
        "cyclomatic",
        "cfg_density",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    return df.sort_values(["episode", "step"]).reset_index(drop=True)


def add_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = df.copy()
    out["scenario"] = scenario
    return out


def summarize_by_step(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (scenario, step), group in df.groupby(["scenario", "step"], sort=True):
        row = {"scenario": scenario, "step": step, "count": len(group)}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_sem"] = (
                row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_with_band(
    ax: plt.Axes,
    summary: pd.DataFrame,
    scenario: str,
    metric: str,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    sub = summary[summary["scenario"] == scenario].sort_values("step")
    x = sub["step"].to_numpy()
    y = sub[f"{metric}_mean"].to_numpy()
    sem = sub[f"{metric}_sem"].to_numpy()
    ci = 1.96 * sem
    ax.plot(x, y, color=color, linewidth=2.2, linestyle=linestyle, label=label)
    ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.15, linewidth=0)


def plot_similarity_decay(
    summary: pd.DataFrame,
    output_path: Path,
    compare_submodels: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    palette = {"Trained": "#8B1E3F", "Random": "#1F4E79"}
    _plot_with_band(ax, summary, "Trained", "mix_sim", palette["Trained"], "SabreL (Trained)")
    _plot_with_band(ax, summary, "Random", "mix_sim", palette["Random"], "Random Policy")

    if compare_submodels:
        for metric, color in [
            ("sub_asm2vec", "#D55E00"),
            ("sub_BinCola", "#009E73"),
            ("sub_CLAP", "#CC79A7"),
            ("sub_jTrans", "#E69F00"),
            ("sub_safe", "#56B4E9"),
        ]:
            if f"{metric}_mean" in summary.columns:
                _plot_with_band(
                    ax,
                    summary,
                    "Trained",
                    metric,
                    color,
                    metric.replace("sub_", ""),
                    linestyle="--",
                )

    ax.set_xlabel("Obfuscation Step")
    ax.set_ylabel("Similarity")
    ax.set_title("Similarity Decay Under a Fixed Obfuscation Budget")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, ncol=2 if compare_submodels else 1, loc="upper right")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    note = (
        "Solid lines: cross-episode mean; shaded regions: 95% confidence intervals."
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_episode_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    sub_cols = [c for c in df.columns if c.startswith("sub_")]

    for (scenario, episode), group in df.groupby(["scenario", "episode"], sort=True):
        group = group.sort_values("step")
        init_row = group.iloc[0]
        final_row = group.iloc[-1]
        mix_vals = group["mix_sim"].to_numpy(dtype=float)

        record = {
            "scenario": scenario,
            "episode": episode,
            "initial_mix": float(init_row["mix_sim"]),
            "final_mix": float(final_row["mix_sim"]),
            "min_mix": float(np.min(mix_vals)),
            "mix_drop": float(init_row["mix_sim"] - final_row["mix_sim"]),
            "min_drop": float(init_row["mix_sim"] - np.min(mix_vals)),
            "steps": int(final_row["step"]),
            "mean_kl": float(group["stealthiness_cost"].mean()),
            "final_kl": float(final_row["stealthiness_cost"]),
            "mean_cyclomatic": float(group["cyclomatic"].mean()),
            "final_cyclomatic": float(final_row["cyclomatic"]),
            "mean_density": float(group["cfg_density"].mean()),
            "final_density": float(final_row["cfg_density"]),
            "trajectory_volatility": float(np.std(np.diff(mix_vals))) if len(mix_vals) > 1 else 0.0,
        }

        for col in sub_cols:
            record[f"{col}_final"] = float(final_row[col])
            record[f"{col}_mean"] = float(group[col].mean())

        records.append(record)

    return pd.DataFrame(records)


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        vals = out[col].to_numpy(dtype=float)
        vmin, vmax = np.min(vals), np.max(vals)
        if np.isclose(vmin, vmax):
            out[col] = 0.5
        else:
            out[col] = (vals - vmin) / (vmax - vmin)
    return out


def plot_behavior_heatmap(features: pd.DataFrame, output_path: Path) -> None:
    feature_order = [
        "mix_drop",
        "min_drop",
        "steps",
        "final_mix",
        "min_mix",
        "mean_kl",
        "final_kl",
        "mean_cyclomatic",
        "final_cyclomatic",
        "mean_density",
        "trajectory_volatility",
    ]
    feature_labels = {
        "mix_drop": "Final Similarity Drop",
        "min_drop": "Best Similarity Drop",
        "steps": "Used Steps",
        "final_mix": "Final Similarity",
        "min_mix": "Minimum Similarity",
        "mean_kl": "Mean KL Cost",
        "final_kl": "Final KL Cost",
        "mean_cyclomatic": "Mean Cyclomatic",
        "final_cyclomatic": "Final Cyclomatic",
        "mean_density": "Mean CFG Density",
        "trajectory_volatility": "Trajectory Volatility",
    }

    available = [f for f in feature_order if f in features.columns]
    scenario_mean = features.groupby("scenario")[available].mean()
    norm = normalize_rows(scenario_mean.T).T
    norm = norm.loc[["Trained", "Random"]]
    norm.columns = [feature_labels[c] for c in norm.columns]

    raw = scenario_mean.loc[["Trained", "Random"]]

    annotations = pd.DataFrame(index=norm.index, columns=norm.columns)
    for scenario in norm.index:
        for raw_col, label in zip(available, norm.columns):
            value = raw.loc[scenario, raw_col]
            annotations.loc[scenario, label] = f"{value:.2f}"

    fig, ax = plt.subplots(figsize=(10.8, 2.9))
    sns.heatmap(
        norm,
        annot=annotations,
        fmt="",
        cmap="YlGnBu",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Row-wise Normalized Intensity"},
        ax=ax,
    )
    ax.set_title("Behavioral Feature Profile of Learned and Random Obfuscation Policies")
    ax.set_xlabel("Trajectory-Level Feature")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_feature_table(features: pd.DataFrame, output_path: Path) -> None:
    summary = features.groupby("scenario").agg(["mean", "std"])
    summary.to_csv(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot academic-style rollout analysis from full_rollout_history.csv files."
    )
    parser.add_argument(
        "--trained",
        type=Path,
        required=True,
        help="Path to the trained full_rollout_history.csv",
    )
    parser.add_argument(
        "--random",
        type=Path,
        required=True,
        help="Path to the random-policy full_rollout_history.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results/analysis_academic"),
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--with-submodels",
        action="store_true",
        help="Overlay trained sub-model similarity curves on the decay plot.",
    )
    parser.add_argument(
        "--trained-episodes",
        type=Path,
        default=None,
        help="Optional path to the trained episodes_history.csv. If omitted, infer it from --trained.",
    )
    parser.add_argument(
        "--random-episodes",
        type=Path,
        default=None,
        help="Optional path to the random episodes_history.csv. If omitted, infer it from --random.",
    )
    return parser.parse_args()


def infer_episodes_csv(full_rollout_csv: Path) -> Path:
    metrics_dir = full_rollout_csv.parent
    return metrics_dir / "episodes_history.csv"


def load_episode_actions(csv_path: Path, scenario: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "actions" not in df.columns:
        raise ValueError(f"Missing 'actions' column in {csv_path}")
    out = df.copy()
    out["scenario"] = scenario
    return out


def summarize_action_behavior(df: pd.DataFrame) -> pd.DataFrame:
    import ast

    action_names = {
        0: "Block Split",
        1: "Opaque Predicate",
        2: "Junk Code",
    }

    rows = []
    for scenario, group in df.groupby("scenario", sort=True):
        counts = {name: 0 for name in action_names.values()}
        lengths = []
        for raw in group["actions"]:
            seq = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
            lengths.append(len(seq))
            for act in seq:
                counts[action_names[int(act)]] += 1

        total = sum(counts.values())
        row = {"scenario": scenario, "mean_actions_per_sample": np.mean(lengths)}
        for name, count in counts.items():
            row[f"{name}_count"] = count
            row[f"{name}_ratio"] = count / total if total > 0 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def plot_strategy_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    ratio_cols = [
        "Block Split_ratio",
        "Opaque Predicate_ratio",
        "Junk Code_ratio",
    ]
    count_cols = [
        "Block Split_count",
        "Opaque Predicate_count",
        "Junk Code_count",
    ]

    heatmap = summary.set_index("scenario")[ratio_cols].copy()
    heatmap.columns = ["Block Split", "Opaque Predicate", "Junk Code"]
    heatmap = heatmap.loc[["Trained", "Random"]]

    counts = summary.set_index("scenario")[count_cols].copy()
    counts.columns = ["Block Split", "Opaque Predicate", "Junk Code"]
    counts = counts.loc[["Trained", "Random"]]

    annotations = pd.DataFrame(index=heatmap.index, columns=heatmap.columns)
    for row in heatmap.index:
        for col in heatmap.columns:
            annotations.loc[row, col] = f"{int(counts.loc[row, col])}\n({heatmap.loc[row, col]:.2f})"

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    sns.heatmap(
        heatmap,
        annot=annotations,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Action Ratio"},
        vmin=0.0,
        vmax=max(0.75, float(np.nanmax(heatmap.to_numpy()))),
        ax=ax,
    )
    ax.set_title("Action Preference Patterns Under Trained and Random Policies")
    ax.set_xlabel("Obfuscation Primitive")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)

    means = summary.set_index("scenario")["mean_actions_per_sample"]
    note = (
        f"Annotation format: count (ratio). "
        f"Mean actions/sample: Trained = {means['Trained']:.2f}, "
        f"Random = {means['Random']:.2f}."
    )
    fig.text(0.5, -0.08, note, ha="center", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()

    trained = add_scenario(load_history(args.trained), "Trained")
    random = add_scenario(load_history(args.random), "Random")
    combined = pd.concat([trained, random], ignore_index=True)

    metrics = ["mix_sim", "stealthiness_cost", "cyclomatic", "cfg_density"]
    metrics.extend([c for c in combined.columns if c.startswith("sub_")])
    summary = summarize_by_step(combined, metrics)
    episode_features = build_episode_features(combined)

    trained_ep_path = args.trained_episodes or infer_episodes_csv(args.trained)
    random_ep_path = args.random_episodes or infer_episodes_csv(args.random)
    trained_actions = load_episode_actions(trained_ep_path, "Trained")
    random_actions = load_episode_actions(random_ep_path, "Random")
    action_summary = summarize_action_behavior(
        pd.concat([trained_actions, random_actions], ignore_index=True)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_similarity_decay(
        summary,
        args.output_dir / "similarity_decay_academic.png",
        compare_submodels=args.with_submodels,
    )
    plot_strategy_heatmap(
        action_summary,
        args.output_dir / "strategy_heatmap_academic.png",
    )
    save_feature_table(
        episode_features,
        args.output_dir / "behavior_feature_summary.csv",
    )
    action_summary.to_csv(args.output_dir / "strategy_action_summary.csv", index=False)

    print(f"[Saved] {args.output_dir / 'similarity_decay_academic.png'}")
    print(f"[Saved] {args.output_dir / 'strategy_heatmap_academic.png'}")
    print(f"[Saved] {args.output_dir / 'behavior_feature_summary.csv'}")
    print(f"[Saved] {args.output_dir / 'strategy_action_summary.csv'}")


if __name__ == "__main__":
    main()
