from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ENTITY_CSV = Path(
    "evaluation_results/Agent_palmtree_200_5-128_Entity_0417H24/metrics/full_rollout_history.csv"
)
SHADOW_CSV = Path(
    "evaluation_results/Agent_palmtree_200_Shadow_0506H34/metrics/full_rollout_history.csv"
)
OUTPUT_STEM = Path("paper_figures/shadow_entity_similarity_comparison")


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        }
    )


def load_comparison_frame(csv_path: Path, mode_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"sub_asm2vec", "sub_CLAP", "sub_safe", "mix_sim"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    out = df[["sub_asm2vec", "sub_CLAP", "sub_safe", "mix_sim"]].copy()
    out = out.rename(
        columns={
            "sub_asm2vec": "asm2vec",
            "sub_CLAP": "CLAP",
            "sub_safe": "SAFE",
            "mix_sim": "Average",
        }
    )
    out["Mode"] = mode_name
    return out


def build_long_frame(entity_csv: Path, shadow_csv: Path) -> pd.DataFrame:
    entity_df = load_comparison_frame(entity_csv, "Real")
    shadow_df = load_comparison_frame(shadow_csv, "Approximate")
    combined = pd.concat([entity_df, shadow_df], ignore_index=True)
    return combined.melt(id_vars=["Mode"], var_name="Model", value_name="Similarity")


def plot_comparison(df_melted: pd.DataFrame, output_stem: Path) -> None:
    configure_style()

    model_order = ["asm2vec", "CLAP", "SAFE", "Average"]
    hue_order = ["Real", "Approximate"]
    palette = {
        "Real": "#8B1E3F",
        "Approximate": "#1F4E79",
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    sns.violinplot(
        data=df_melted,
        x="Model",
        y="Similarity",
        hue="Mode",
        order=model_order,
        hue_order=hue_order,
        palette=palette,
        cut=0,
        inner=None,
        linewidth=0.9,
        saturation=0.9,
        width=0.82,
        alpha=0.22,
        ax=ax,
    )

    sns.boxplot(
        data=df_melted,
        x="Model",
        y="Similarity",
        hue="Mode",
        order=model_order,
        hue_order=hue_order,
        dodge=True,
        palette=palette,
        width=0.24,
        linewidth=1.0,
        fliersize=0,
        saturation=1.0,
        boxprops={"zorder": 3},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        medianprops={"color": "#222222", "linewidth": 1.2},
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[: len(hue_order)],
        labels[: len(hue_order)],
        title=None,
        frameon=False,
        ncol=1,
        loc="lower right",
        handlelength=1.6,
    )

    ax.set_xlabel("Similarity Model")
    ax.set_ylabel("Similarity Score")
    ax.set_title("Real and Approximate Training Produce Similar Similarity Profiles", pad=10)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", which="major", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.grid(True, axis="y", which="minor", color="#eeeeee", linewidth=0.5, alpha=0.8)
    ax.grid(False, axis="x")
    ax.minorticks_on()

    # note = (
    #     "Violins show the full distribution; boxplots summarize quartiles and medians."
    # )
    # fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    melted = build_long_frame(ENTITY_CSV, SHADOW_CSV)
    plot_comparison(melted, OUTPUT_STEM)
    print(f"[*] Saved {OUTPUT_STEM.with_suffix('.svg')}")
    print(f"[*] Saved {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"[*] Saved {OUTPUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
