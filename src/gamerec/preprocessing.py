"""
GameRec Preprocessing & Data Transformation Pipeline.
Handles cleaning, deduplication, schema downcasting, Bayesian shrinkage rating,
and benchmark extraction for memory-efficient recommendation model training.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gamerec.preprocessing")


def clean_games_catalog(df_games: pl.DataFrame, df_meta: pl.DataFrame) -> pl.DataFrame:
    """
    Merges games.csv and games_metadata.json into a unified, downcasted game catalog.
    Handles empty descriptions and tags cleanly without data loss.
    """
    logger.info("Cleaning games catalog and joining metadata...")
    if df_games.shape[0] != 50872 or df_meta.shape[0] != 50872:
        logger.warning(
            f"Expected 50,872 games; found games={df_games.shape[0]}, meta={df_meta.shape[0]}"
        )

    # Join on app_id
    merged = df_games.join(df_meta, on="app_id", how="left")

    # Bayesian rating prior calculation (Shrinkage estimator)
    # W = (v / (v + m)) * R + (m / (v + m)) * C
    # where R = positive_ratio / 100, v = user_reviews, C = mean positive ratio, m = 75th percentile of user_reviews
    mean_positive_ratio = merged["positive_ratio"].mean() / 100.0
    # Threshold m: 75th percentile of user reviews to regularize games with very few reviews
    m_threshold = float(merged["user_reviews"].quantile(0.75))

    # Clean description and tags
    cleaned = (
        merged
        .with_columns([
            pl.col("app_id").cast(pl.UInt32),
            pl.col("title").str.strip_chars(),
            pl.col("positive_ratio").cast(pl.UInt8),
            pl.col("user_reviews").cast(pl.UInt32),
            pl.col("price_final").cast(pl.Float32),
            pl.col("price_original").cast(pl.Float32),
            pl.col("discount").cast(pl.Float32),
            pl.col("win").cast(pl.Boolean),
            pl.col("mac").cast(pl.Boolean),
            pl.col("linux").cast(pl.Boolean),
            pl.col("steam_deck").cast(pl.Boolean),
            # Description handling
            pl.col("description").fill_null("").str.strip_chars().alias("description"),
            (pl.col("description").fill_null("").str.strip_chars() != "").alias("has_description"),
            # Tags handling
            pl.col("tags").fill_null(pl.lit([])).alias("tags"),
            (pl.col("tags").list.len() > 0).alias("has_tags"),
            # Year extraction
            pl.col("date_release").str.slice(0, 4).cast(pl.Int32, strict=False).alias("release_year"),
            # Price tier
            (pl.col("price_final") == 0.0).alias("is_free"),
        ])
        .with_columns([
            # Flatten tags into space-separated string for TF-IDF
            pl.col("tags").list.join(" ").alias("tags_str"),
            # Compute Bayesian weighted score (0.0 to 1.0)
            (
                (
                    (pl.col("user_reviews").cast(pl.Float64) / (pl.col("user_reviews").cast(pl.Float64) + m_threshold))
                    * (pl.col("positive_ratio").cast(pl.Float64) / 100.0)
                )
                + (
                    (m_threshold / (pl.col("user_reviews").cast(pl.Float64) + m_threshold))
                    * mean_positive_ratio
                )
            ).cast(pl.Float32).alias("bayesian_rating"),
        ])
    )

    # Validate output
    assert cleaned["app_id"].n_unique() == cleaned.shape[0], "Duplicate app_id detected in catalog!"
    assert cleaned["app_id"].is_null().sum() == 0, "Null app_id detected in catalog!"
    logger.info(f"Successfully processed games catalog: {cleaned.shape[0]:,} games.")
    return cleaned


def inspect_duplicate_interactions(scan_rec: pl.LazyFrame) -> pl.DataFrame:
    """
    Finds and returns any repeated (user_id, app_id) interactions.
    """
    logger.info("Checking for duplicate (user_id, app_id) interaction pairs...")
    dup_pairs = (
        scan_rec
        .group_by(["user_id", "app_id"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .collect()
    )
    logger.info(f"Found {dup_pairs.shape[0]} duplicate (user_id, app_id) pairs.")
    return dup_pairs


def clean_interactions(
    scan_rec: pl.LazyFrame,
    deduplicate: bool = True,
) -> pl.LazyFrame:
    """
    Cleans raw recommendations with schema downcasting, timestamp parsing,
    and resolution of repeated interactions.
    
    Feedback semantics:
    - Explicit positive feedback: is_recommended == True (feedback_type = 1)
    - Explicit negative feedback: is_recommended == False (feedback_type = -1)
    - Missing feedback: unobserved interactions (NEVER imputed as negative)
    - Implicit confidence: ln(1.0 + hours)
    """
    logger.info("Constructing cleaned interactions LazyFrame...")

    # Type casting and feature engineering
    processed = (
        scan_rec
        .with_columns([
            pl.col("app_id").cast(pl.UInt32),
            pl.col("user_id").cast(pl.UInt32),
            pl.col("review_id").cast(pl.UInt32),
            pl.col("is_recommended").cast(pl.Boolean),
            # Hours clamped between 0 and 1000
            pl.col("hours").cast(pl.Float32).clip(lower_bound=0.0, upper_bound=1000.0).alias("hours"),
            pl.col("date").str.to_date("%Y-%m-%d", strict=False).alias("date"),
            pl.col("helpful").cast(pl.UInt32),
            pl.col("funny").cast(pl.UInt32),
            # Explicit feedback flag
            pl.when(pl.col("is_recommended"))
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(-1, dtype=pl.Int8))
            .alias("feedback_type"),
            # Implicit confidence weight: ln(1 + hours)
            (pl.col("hours") + 1.0).log().cast(pl.Float32).alias("implicit_confidence"),
        ])
    )

    if deduplicate:
        # Sort so that the latest review (by date, then hours, then review_id) is kept
        processed = (
            processed
            .sort(["user_id", "app_id", "date", "hours", "review_id"], descending=False)
            .unique(subset=["user_id", "app_id"], keep="last")
        )

    return processed


def create_active_benchmark(
    scan_clean: pl.LazyFrame,
    min_user_reviews: int = 10,
    min_game_reviews: int = 20,
    max_users: Optional[int] = 100_000,
    random_seed: int = 42,
) -> pl.DataFrame:
    """
    Filters interactions to a dense active-user k-core benchmark subset.
    Ensures sufficient user history for leak-free temporal evaluation (train/val/test splits)
    and prevents out-of-memory errors during collaborative filtering matrix decomposition.
    """
    logger.info(
        f"Creating active benchmark: min_user_reviews={min_user_reviews}, min_game_reviews={min_game_reviews}, max_users={max_users}..."
    )

    # 1. Filter users with >= min_user_reviews
    user_counts = (
        scan_clean
        .group_by("user_id")
        .agg(pl.len().alias("user_review_count"))
        .filter(pl.col("user_review_count") >= min_user_reviews)
    )

    if max_users is not None:
        # Sample deterministically
        user_list_df = user_counts.collect()
        if user_list_df.shape[0] > max_users:
            logger.info(f"Sampling {max_users:,} users out of {user_list_df.shape[0]:,} qualified users...")
            user_list_df = user_list_df.sample(n=max_users, seed=random_seed)
        eligible_user_ids = user_list_df["user_id"]
        filtered_by_user = scan_clean.filter(pl.col("user_id").is_in(eligible_user_ids))
    else:
        filtered_by_user = scan_clean.join(user_counts, on="user_id", how="inner").drop("user_review_count")

    # 2. Filter games with >= min_game_reviews
    game_counts = (
        filtered_by_user
        .group_by("app_id")
        .agg(pl.len().alias("game_review_count"))
        .filter(pl.col("game_review_count") >= min_game_reviews)
    )

    benchmark_df = (
        filtered_by_user
        .join(game_counts, on="app_id", how="inner")
        .drop("game_review_count")
        .collect()
        .sort(["user_id", "date", "review_id"])
    )

    logger.info(
        f"Active benchmark created: {benchmark_df.shape[0]:,} interactions, "
        f"{benchmark_df['user_id'].n_unique():,} unique users, "
        f"{benchmark_df['app_id'].n_unique():,} unique games."
    )
    return benchmark_df


def compute_dataset_metrics(
    catalog_df: pl.DataFrame,
    interactions_lazy: pl.LazyFrame,
) -> Dict[str, Any]:
    """
    Generates exact data-quality statistics and matrix sparsity measurements.
    """
    logger.info("Computing exact data quality metrics...")
    
    total_games_catalog = catalog_df.shape[0]
    catalog_app_ids = set(catalog_df["app_id"].to_list())

    rec_stats = interactions_lazy.select([
        pl.len().alias("total_interactions"),
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("app_id").n_unique().alias("unique_games"),
        (pl.col("is_recommended") == True).sum().alias("pos_interactions"),
        (pl.col("is_recommended") == False).sum().alias("neg_interactions"),
        pl.col("hours").min().alias("min_hours"),
        pl.col("hours").median().alias("median_hours"),
        pl.col("hours").mean().alias("mean_hours"),
        pl.col("hours").quantile(0.95).alias("p95_hours"),
        pl.col("hours").max().alias("max_hours"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ]).collect().to_dicts()[0]

    unique_users = rec_stats["unique_users"]
    unique_games_rec = rec_stats["unique_games"]
    total_interactions = rec_stats["total_interactions"]

    # Sparsity calculation
    theoretical_max_interactions = float(unique_users) * float(unique_games_rec)
    sparsity = (1.0 - (float(total_interactions) / theoretical_max_interactions)) * 100.0

    # User degree distribution
    user_degrees = (
        interactions_lazy
        .group_by("user_id")
        .agg(pl.len().alias("degree"))
        .select([
            pl.col("degree").min().alias("min_deg"),
            pl.col("degree").quantile(0.25).alias("p25_deg"),
            pl.col("degree").median().alias("median_deg"),
            pl.col("degree").mean().alias("mean_deg"),
            pl.col("degree").quantile(0.75).alias("p75_deg"),
            pl.col("degree").quantile(0.95).alias("p95_deg"),
            pl.col("degree").max().alias("max_deg"),
        ])
        .collect()
        .to_dicts()[0]
    )

    # Game degree distribution
    game_degrees = (
        interactions_lazy
        .group_by("app_id")
        .agg(pl.len().alias("degree"))
        .select([
            pl.col("degree").min().alias("min_deg"),
            pl.col("degree").quantile(0.25).alias("p25_deg"),
            pl.col("degree").median().alias("median_deg"),
            pl.col("degree").mean().alias("mean_deg"),
            pl.col("degree").quantile(0.75).alias("p75_deg"),
            pl.col("degree").quantile(0.95).alias("p95_deg"),
            pl.col("degree").max().alias("max_deg"),
        ])
        .collect()
        .to_dicts()[0]
    )

    # Join coverage
    rec_apps = set(interactions_lazy.select(pl.col("app_id").unique()).collect()["app_id"].to_list())
    orphan_apps = rec_apps - catalog_app_ids
    unreviewed_catalog_games = catalog_app_ids - rec_apps

    metrics = {
        "catalog": {
            "total_games": total_games_catalog,
            "unique_app_ids": catalog_df["app_id"].n_unique(),
            "games_with_description": catalog_df["has_description"].sum(),
            "games_missing_description": (~catalog_df["has_description"]).sum(),
            "games_with_tags": catalog_df["has_tags"].sum(),
            "games_missing_tags": (~catalog_df["has_tags"]).sum(),
            "unreviewed_catalog_games": len(unreviewed_catalog_games),
            "unreviewed_ratio_pct": round(len(unreviewed_catalog_games) / total_games_catalog * 100.0, 2),
        },
        "interactions": {
            "total_records": total_interactions,
            "unique_users": unique_users,
            "unique_games": unique_games_rec,
            "orphan_games": len(orphan_apps),
            "positive_count": rec_stats["pos_interactions"],
            "positive_ratio_pct": round(rec_stats["pos_interactions"] / total_interactions * 100.0, 2),
            "negative_count": rec_stats["neg_interactions"],
            "negative_ratio_pct": round(rec_stats["neg_interactions"] / total_interactions * 100.0, 2),
            "date_range": [str(rec_stats["min_date"]), str(rec_stats["max_date"])],
            "playtime_stats": {
                "min": float(rec_stats["min_hours"]),
                "median": float(rec_stats["median_hours"]),
                "mean": round(float(rec_stats["mean_hours"]), 2),
                "p95": round(float(rec_stats["p95_hours"]), 2),
                "max": float(rec_stats["max_hours"]),
            },
            "matrix_sparsity_pct": round(sparsity, 5),
            "user_interactions_per_user": user_degrees,
            "game_interactions_per_game": game_degrees,
        },
    }
    return metrics


def run_preprocessing_pipeline(
    raw_dir: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete Phase 2 preprocessing workflow:
    1. Ingests raw games and metadata
    2. Builds and validates games_catalog.parquet
    3. Cleans, deduplicates, and downcasts recommendations
    4. Extracts active benchmark dataset for collaborative filtering and evaluation
    5. Saves all processed artifacts to disk
    """
    from gamerec.data import (
        get_default_paths,
        load_raw_games,
        load_raw_metadata,
        load_raw_recommendations_lazy,
    )

    paths = get_default_paths()
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
    else:
        raw_dir = paths["raw_dir"]

    if processed_dir is not None:
        processed_dir = Path(processed_dir)
    else:
        processed_dir = paths["processed_dir"]

    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 2 data preprocessing pipeline...")

    # Step 1: Catalog Processing
    df_raw_games = load_raw_games(raw_dir / "games.csv")
    df_raw_meta = load_raw_metadata(raw_dir / "games_metadata.json")
    catalog_clean = clean_games_catalog(df_raw_games, df_raw_meta)

    catalog_out = processed_dir / "games_catalog.parquet"
    catalog_clean.write_parquet(catalog_out, compression="zstd")
    logger.info(f"Saved clean games catalog to {catalog_out} ({round(catalog_out.stat().st_size / (1024*1024), 2)} MB)")

    # Step 2: Interactions Processing
    scan_raw_rec = load_raw_recommendations_lazy(raw_dir / "recommendations.csv")
    scan_clean_rec = clean_interactions(scan_raw_rec, deduplicate=True)

    # Step 3: Create Active Benchmark Dataset (10-core active users, sampled to 100k users)
    # This provides a realistic ~1.5M interaction dataset with complete temporal evaluation integrity
    benchmark_df = create_active_benchmark(
        scan_clean_rec,
        min_user_reviews=10,
        min_game_reviews=20,
        max_users=100_000,
        random_seed=42,
    )

    benchmark_out = processed_dir / "interactions_benchmark.parquet"
    benchmark_df.write_parquet(benchmark_out, compression="zstd")
    logger.info(f"Saved active benchmark interactions to {benchmark_out} ({round(benchmark_out.stat().st_size / (1024*1024), 2)} MB)")

    # Step 4: Compute comprehensive dataset metrics
    metrics = compute_dataset_metrics(catalog_clean, scan_clean_rec)

    metrics_out = processed_dir / "dataset_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics summary to {metrics_out}")

    return {
        "catalog_path": str(catalog_out),
        "benchmark_path": str(benchmark_out),
        "metrics_path": str(metrics_out),
        "metrics": metrics,
    }
