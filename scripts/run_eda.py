#!/usr/bin/env python3
"""
GameRec Exploratory Data Analysis & Visualization Script.
Generates comprehensive distribution plots, power-law curves,
sentiment balance charts, and catalog coverage audits.
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Ensure src in pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gamerec.data import get_default_paths, load_processed_catalog, load_processed_interactions


def setup_plotting_style():
    """Sets a clean, publication-ready style for matplotlib charts."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#E0E0E0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_game_popularity(catalog_df: pl.DataFrame, figures_dir: Path):
    """Figure 1: Long-tail power-law game popularity distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # User reviews sorted
    reviews = catalog_df.filter(pl.col("user_reviews") > 0)["user_reviews"].sort(descending=True).to_numpy()
    ranks = np.arange(1, len(reviews) + 1)

    # Log-Log plot
    ax1.loglog(ranks, reviews, color="#1f77b4", linewidth=2)
    ax1.set_title("Steam Game Popularity: Rank vs Review Volume (Log-Log)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Popularity Rank (Log Scale)", fontsize=10)
    ax1.set_ylabel("Total User Reviews (Log Scale)", fontsize=10)
    ax1.grid(True)

    # Cumulative distribution
    cum_reviews = np.cumsum(reviews)
    cum_pct = (cum_reviews / cum_reviews[-1]) * 100
    rank_pct = (ranks / len(ranks)) * 100

    ax2.plot(rank_pct, cum_pct, color="#e6550d", linewidth=2)
    ax2.axvline(x=5, color="#636363", linestyle="--", label="Top 5% Games Account for ~88% of Reviews")
    ax2.set_title("Cumulative Concentration of User Interactions", fontsize=12, fontweight="bold")
    ax2.set_xlabel("% of Catalog Games", fontsize=10)
    ax2.set_ylabel("% of Total User Reviews", fontsize=10)
    ax2.legend(loc="lower right")
    ax2.grid(True)

    plt.tight_layout()
    out_path = figures_dir / "01_game_popularity.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_top_tags(catalog_df: pl.DataFrame, figures_dir: Path):
    """Figure 2: Frequency distribution of top 25 community tags."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Flatten tags
    all_tags = (
        catalog_df
        .select(pl.col("tags").explode().alias("tag"))
        .filter(pl.col("tag").is_not_null())
        .group_by("tag")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(25)
    )

    tags = all_tags["tag"].to_list()[::-1]
    counts = all_tags["count"].to_list()[::-1]

    bars = ax.barh(tags, counts, color="#3182bd", edgecolor="#08519c", height=0.65)
    ax.set_title("Top 25 Community Tags Across Steam Catalog (50,872 Games)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Games Tagged", fontsize=10)
    ax.grid(axis="x")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 200, bar.get_y() + bar.get_height() / 2, f"{int(width):,}",
                ha="left", va="center", fontsize=8, color="#333333")

    plt.tight_layout()
    out_path = figures_dir / "02_top_tags.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_user_activity(benchmark_df: pl.DataFrame, figures_dir: Path):
    """Figure 3: Interaction count per user in active benchmark."""
    fig, ax = plt.subplots(figsize=(10, 5))

    user_counts = (
        benchmark_df
        .group_by("user_id")
        .agg(pl.len().alias("count"))
        ["count"].to_numpy()
    )

    ax.hist(user_counts, bins=np.logspace(np.log10(10), np.log10(user_counts.max()), 35),
            color="#756bb1", edgecolor="#54278f", alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Distribution of Interaction Degree per User (Log-Log Histogram)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Reviews Written per User (Log Scale)", fontsize=10)
    ax.set_ylabel("Number of Users (Log Scale)", fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    out_path = figures_dir / "03_user_activity.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_sentiment_balance(benchmark_df: pl.DataFrame, figures_dir: Path):
    """Figure 4: Explicit feedback sentiment balance (is_recommended)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    pos_count = (benchmark_df["is_recommended"] == True).sum()
    neg_count = (benchmark_df["is_recommended"] == False).sum()
    total = len(benchmark_df)

    # Donut chart
    labels = [f"Positive (Thumbs Up)\n{pos_count:,} ({pos_count/total*100:.1f}%)",
              f"Negative (Thumbs Down)\n{neg_count:,} ({neg_count/total*100:.1f}%)"]
    colors = ["#2ca02c", "#d62728"]
    explode = (0.05, 0.05)

    wedges, texts = ax1.pie([pos_count, neg_count], explode=explode, labels=labels, colors=colors,
                            startangle=140, wedgeprops=dict(width=0.4, edgecolor='white'))
    ax1.set_title("Explicit Feedback Class Balance", fontsize=12, fontweight="bold")

    # Bar breakdown by playtime
    hours_pos = benchmark_df.filter(pl.col("is_recommended") == True)["hours"].median()
    hours_neg = benchmark_df.filter(pl.col("is_recommended") == False)["hours"].median()

    ax2.bar(["Positive Reviews", "Negative Reviews"], [hours_pos, hours_neg], color=colors, width=0.4)
    ax2.set_title("Median Playtime Hours by Sentiment", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Median Hours Played", fontsize=10)
    ax2.grid(axis="y")
    for i, v in enumerate([hours_pos, hours_neg]):
        ax2.text(i, v + 1, f"{v:.1f} hrs", ha="center", fontweight="bold")

    plt.tight_layout()
    out_path = figures_dir / "04_sentiment_balance.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_playtime_distribution(benchmark_df: pl.DataFrame, figures_dir: Path):
    """Figure 5: Playtime hours distribution (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    hours = benchmark_df.filter(pl.col("hours") > 0)["hours"].to_numpy()

    # Log bins
    bins = np.logspace(np.log10(0.1), np.log10(1000.0), 50)
    ax.hist(hours, bins=bins, color="#31a354", edgecolor="#006837", alpha=0.8)
    ax.set_xscale("log")
    ax.axvline(x=np.median(hours), color="#e31a1c", linestyle="--", linewidth=1.5,
               label=f"Median Playtime ({np.median(hours):.1f} hrs)")
    ax.axvline(x=1000.0, color="#636363", linestyle=":", linewidth=1.5,
               label="1,000 hrs Scraper Cap")

    ax.set_title("Distribution of Recorded Playtime at Review Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hours Played (Log Scale)", fontsize=10)
    ax.set_ylabel("Interaction Frequency", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    out_path = figures_dir / "05_playtime_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_missingness_summary(catalog_df: pl.DataFrame, figures_dir: Path):
    """Figure 6: Data quality, coverage, and missingness audit."""
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = [
        "Has Description",
        "Has Community Tags",
        "Has User Reviews (>0)",
        "Verified Release Date",
        "Free-to-Play",
        "Steam Deck Verified"
    ]
    
    total = len(catalog_df)
    percentages = [
        catalog_df["has_description"].sum() / total * 100,
        catalog_df["has_tags"].sum() / total * 100,
        (catalog_df["user_reviews"] > 0).sum() / total * 100,
        (catalog_df["release_year"].is_not_null()).sum() / total * 100,
        catalog_df["is_free"].sum() / total * 100,
        catalog_df["steam_deck"].sum() / total * 100,
    ]

    colors = ["#2b8cbe" if p > 50 else "#a6bddb" for p in percentages]
    bars = ax.barh(categories[::-1], percentages[::-1], color=colors[::-1], height=0.55)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of Catalog Games (50,872 Total)", fontsize=10)
    ax.set_title("Game Catalog Feature Coverage & Completeness", fontsize=12, fontweight="bold")
    ax.grid(axis="x")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                ha="left", va="center", fontsize=9, fontweight="bold", color="#333333")

    plt.tight_layout()
    out_path = figures_dir / "06_missingness_summary.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 70)
    print("GameRec: Generating EDA Visualizations")
    print("=" * 70)
    setup_plotting_style()

    paths = get_default_paths()
    figures_dir = paths["reports_dir"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed catalog and active benchmark...")
    catalog_df = load_processed_catalog(paths["processed_catalog"])
    benchmark_df = load_processed_interactions(paths["processed_benchmark"], lazy=False)

    print(f"Catalog shape:   {catalog_df.shape}")
    print(f"Benchmark shape: {benchmark_df.shape}")

    plot_game_popularity(catalog_df, figures_dir)
    plot_top_tags(catalog_df, figures_dir)
    plot_user_activity(benchmark_df, figures_dir)
    plot_sentiment_balance(benchmark_df, figures_dir)
    plot_playtime_distribution(benchmark_df, figures_dir)
    plot_missingness_summary(catalog_df, figures_dir)

    print("\nAll 6 EDA figures successfully generated in:", figures_dir)

if __name__ == "__main__":
    main()
