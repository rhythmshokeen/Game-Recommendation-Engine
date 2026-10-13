#!/usr/bin/env python3
"""
GameRec Dataset Inspection & Profiling Script
Performs memory-efficient schema, integrity, and leakage profiling on raw Steam datasets.
"""

import os
import sys
import json
import time
from pathlib import Path

def get_file_stats(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "inode": stat.st_ino,
        "is_symlink": path.is_symlink(),
    }

def main():
    start_time = time.time()
    print("=" * 70)
    print("GameRec: Steam Dataset Inspection & Integrity Verification")
    print("=" * 70)

    # Use Polars for zero-copy streaming and memory efficiency
    try:
        import polars as pl
    except ImportError:
        print("Polars not installed. Please install polars.")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    data_raw = project_root / "data" / "raw"
    dataset_dir = project_root / "dataset"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"data/raw: {data_raw}")
    print(f"dataset:  {dataset_dir}")

    # 1. Compare directory contents & inode identity
    print("\n[1] Comparing directories (dataset/ vs data/raw/)...")
    dir_comparison = {}
    for fname in ["games.csv", "games_metadata.json", "recommendations.csv", "users.csv"]:
        f_raw = data_raw / fname
        f_set = dataset_dir / fname
        
        stat_raw = get_file_stats(f_raw) if f_raw.exists() else None
        stat_set = get_file_stats(f_set) if f_set.exists() else None
        
        identical_inode = (stat_raw["inode"] == stat_set["inode"]) if (stat_raw and stat_set) else False
        dir_comparison[fname] = {
            "data_raw": stat_raw,
            "dataset": stat_set,
            "identical_physical_file": identical_inode
        }
        print(f"  - {fname}: Raw Size: {stat_raw['size_mb']}MB | Dataset Size: {stat_set['size_mb']}MB | Inode match: {identical_inode}")

    # 2. Inspect games.csv
    print("\n[2] Inspecting games.csv...")
    games_path = data_raw / "games.csv"
    df_games = pl.read_csv(games_path)
    games_rows, games_cols = df_games.shape
    games_nulls = {col: df_games[col].is_null().sum() for col in df_games.columns}
    games_dtypes = {col: str(dtype) for col, dtype in zip(df_games.columns, df_games.dtypes)}
    games_unique_apps = df_games["app_id"].n_unique()

    print(f"  Shape: {games_rows:,} rows x {games_cols} cols")
    print(f"  Unique app_id: {games_unique_apps:,}")
    print(f"  Null counts: {games_nulls}")

    # 3. Inspect games_metadata.json
    print("\n[3] Inspecting games_metadata.json...")
    meta_path = data_raw / "games_metadata.json"
    meta_line_count = 0
    meta_app_ids = set()
    empty_desc_count = 0
    empty_tags_count = 0
    all_tags = set()
    
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta_line_count += 1
            record = json.loads(line)
            aid = record.get("app_id")
            if aid is not None:
                meta_app_ids.add(aid)
            desc = record.get("description", "")
            if not desc or desc.strip() == "":
                empty_desc_count += 1
            tags = record.get("tags", [])
            if not tags:
                empty_tags_count += 1
            else:
                all_tags.update(tags)

    print(f"  Total lines: {meta_line_count:,}")
    print(f"  Unique app_ids: {len(meta_app_ids):,}")
    print(f"  Overlap with games.csv: {len(set(df_games['app_id'].to_list()).intersection(meta_app_ids)):,} / {games_unique_apps:,}")
    print(f"  Empty descriptions: {empty_desc_count:,} ({empty_desc_count / meta_line_count * 100:.2f}%)")
    print(f"  Empty tags: {empty_tags_count:,} ({empty_tags_count / meta_line_count * 100:.2f}%)")
    print(f"  Vocabulary of tags: {len(all_tags):,} unique tags")

    # 4. Inspect users.csv
    print("\n[4] Inspecting users.csv (streaming/lazy)...")
    users_path = data_raw / "users.csv"
    scan_users = pl.scan_csv(users_path)
    users_schema = {col: str(dt) for col, dt in scan_users.collect_schema().items()}
    users_count = scan_users.select(pl.len()).collect().item()
    users_nulls = scan_users.null_count().collect().to_dicts()[0]
    users_stats = scan_users.select([
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("products").min().alias("min_products"),
        pl.col("products").median().alias("median_products"),
        pl.col("products").mean().alias("mean_products"),
        pl.col("products").max().alias("max_products"),
        pl.col("reviews").min().alias("min_reviews"),
        pl.col("reviews").median().alias("median_reviews"),
        pl.col("reviews").mean().alias("mean_reviews"),
        pl.col("reviews").max().alias("max_reviews"),
    ]).collect().to_dicts()[0]

    print(f"  Total users: {users_count:,} | Unique: {users_stats['unique_users']:,}")
    print(f"  Null counts: {users_nulls}")
    print(f"  Products: median={users_stats['median_products']}, mean={users_stats['mean_products']:.1f}, max={users_stats['max_products']:,}")
    print(f"  Reviews authored: median={users_stats['median_reviews']}, mean={users_stats['mean_reviews']:.2f}, max={users_stats['max_reviews']:,}")

    # 5. Inspect recommendations.csv (lazy scan)
    print("\n[5] Inspecting recommendations.csv (lazy streaming)...")
    rec_path = data_raw / "recommendations.csv"
    scan_rec = pl.scan_csv(rec_path)
    rec_schema = {col: str(dt) for col, dt in scan_rec.collect_schema().items()}
    rec_count = scan_rec.select(pl.len()).collect().item()
    rec_nulls = scan_rec.null_count().collect().to_dicts()[0]
    
    print(f"  Total records: {rec_count:,}")
    print(f"  Schema: {rec_schema}")
    print(f"  Null counts: {rec_nulls}")

    # Metrics and distributions
    rec_stats = scan_rec.select([
        pl.col("app_id").n_unique().alias("unique_apps"),
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("review_id").n_unique().alias("unique_reviews"),
        (pl.col("is_recommended") == True).mean().alias("positive_rec_ratio"),
        pl.col("hours").min().alias("min_hours"),
        pl.col("hours").median().alias("median_hours"),
        pl.col("hours").mean().alias("mean_hours"),
        pl.col("hours").quantile(0.95).alias("p95_hours"),
        pl.col("hours").max().alias("max_hours"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("helpful").max().alias("max_helpful"),
        pl.col("funny").max().alias("max_funny"),
    ]).collect().to_dicts()[0]

    print(f"  Unique games in recs: {rec_stats['unique_apps']:,} / {games_unique_apps:,}")
    print(f"  Unique users in recs: {rec_stats['unique_users']:,} / {users_count:,}")
    print(f"  Unique review_ids: {rec_stats['unique_reviews']:,} (matches total: {rec_stats['unique_reviews'] == rec_count})")
    print(f"  Positive recommendation ratio: {rec_stats['positive_rec_ratio'] * 100:.2f}%")
    print(f"  Playtime hours: min={rec_stats['min_hours']}, median={rec_stats['median_hours']}, mean={rec_stats['mean_hours']:.1f}, p95={rec_stats['p95_hours']:.1f}, max={rec_stats['max_hours']}")
    print(f"  Date range: {rec_stats['min_date']} to {rec_stats['max_date']}")

    # 6. Check Join Key Integrity
    print("\n[6] Join Key & Referential Integrity Verification...")
    # Check if all rec app_ids exist in games.csv
    rec_app_ids = scan_rec.select(pl.col("app_id").unique()).collect()["app_id"].to_list()
    rec_app_set = set(rec_app_ids)
    games_app_set = set(df_games["app_id"].to_list())
    
    orphan_rec_apps = rec_app_set - games_app_set
    catalog_unreviewed_games = games_app_set - rec_app_set
    print(f"  Join Key between games and recs: 'app_id' (Int64)")
    print(f"  Rec games missing in games.csv (orphans): {len(orphan_rec_apps)}")
    print(f"  Catalog games with 0 recommendations (cold-start games): {len(catalog_unreviewed_games):,} ({len(catalog_unreviewed_games)/games_unique_apps*100:.2f}%)")

    # 7. Duplicate Checks
    print("\n[7] Duplicate Interaction Check...")
    duplicate_user_app = (
        scan_rec
        .group_by(["user_id", "app_id"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    print(f"  Duplicate (user_id, app_id) interaction pairs: {duplicate_user_app} out of {rec_count:,}")

    # 8. User Interaction Sparsity Analysis
    print("\n[8] User Interaction Distribution...")
    user_interaction_summary = (
        scan_rec
        .group_by("user_id")
        .agg(pl.len().alias("n_reviews"))
        .select([
            (pl.col("n_reviews") == 1).sum().alias("users_with_1"),
            ((pl.col("n_reviews") >= 2) & (pl.col("n_reviews") < 5)).sum().alias("users_2_to_4"),
            (pl.col("n_reviews") >= 5).sum().alias("users_gte_5"),
            (pl.col("n_reviews") >= 10).sum().alias("users_gte_10"),
            (pl.col("n_reviews") >= 20).sum().alias("users_gte_20"),
        ])
        .collect()
        .to_dicts()[0]
    )
    print(f"  Users with 1 review:  {user_interaction_summary['users_with_1']:,} ({user_interaction_summary['users_with_1'] / rec_stats['unique_users'] * 100:.1f}%)")
    print(f"  Users with 2-4 recs:  {user_interaction_summary['users_2_to_4']:,} ({user_interaction_summary['users_2_to_4'] / rec_stats['unique_users'] * 100:.1f}%)")
    print(f"  Users with >= 5 recs: {user_interaction_summary['users_gte_5']:,} ({user_interaction_summary['users_gte_5'] / rec_stats['unique_users'] * 100:.1f}%)")
    print(f"  Users with >= 10 recs:{user_interaction_summary['users_gte_10']:,} ({user_interaction_summary['users_gte_10'] / rec_stats['unique_users'] * 100:.1f}%)")
    print(f"  Users with >= 20 recs:{user_interaction_summary['users_gte_20']:,} ({user_interaction_summary['users_gte_20'] / rec_stats['unique_users'] * 100:.1f}%)")

    # 9. Potential Data Leakage Risks Identification
    print("\n[9] Data Leakage Assessment...")
    leakage_notes = [
        "1. Temporal Leakage: If a standard random train/test split is used across interactions, the model would train on future user actions to predict past actions. Evaluation MUST use temporal holdout (splitting by interaction date or holding out each user's latest interaction).",
        "2. Future Meta-feedback Leakage: Features 'helpful' and 'funny' accumulate over the entire lifetime of a review and reflect social feedback received long after the interaction occurred. These must NEVER be used as model inputs to predict recommendations.",
        "3. Playtime (hours) Target Leakage: 'hours' recorded in the review reflects total accumulated playtime at the time the review was written, not at recommendation time. However, it serves as a valid implicit confidence weight for collaborative filtering (log1p(hours)).",
        "4. Cold-Start Leakage: Evaluating collaborative filtering on users/items with 0 training interactions without fallback can artificially inflate failure rates or cause errors; models must be evaluated with explicit cold-start protocols."
    ]
    for note in leakage_notes:
        print(f"  * {note}")

    # Compile comprehensive report data
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_sec": round(time.time() - start_time, 2),
        "source": {
            "name": "Game Recommendations on Steam",
            "author": "Anton Kozyriev",
            "platform": "Kaggle",
            "license": "CC0: Public Domain",
            "url": "https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam"
        },
        "directory_comparison": dir_comparison,
        "games_csv": {
            "rows": games_rows,
            "cols": games_cols,
            "unique_app_id": games_unique_apps,
            "dtypes": games_dtypes,
            "nulls": games_nulls,
        },
        "games_metadata_json": {
            "lines": meta_line_count,
            "unique_app_id": len(meta_app_ids),
            "empty_descriptions": empty_desc_count,
            "empty_tags": empty_tags_count,
            "vocab_tags_count": len(all_tags),
        },
        "users_csv": {
            "rows": users_count,
            "unique_users": users_stats["unique_users"],
            "schema": users_schema,
            "nulls": users_nulls,
            "stats": users_stats,
        },
        "recommendations_csv": {
            "rows": rec_count,
            "schema": rec_schema,
            "nulls": rec_nulls,
            "stats": rec_stats,
            "duplicate_interactions": duplicate_user_app,
            "user_sparsity": user_interaction_summary,
        },
        "join_keys": {
            "games_to_recommendations": "app_id",
            "users_to_recommendations": "user_id",
            "orphan_recommendations_apps": len(orphan_rec_apps),
            "unreviewed_catalog_games": len(catalog_unreviewed_games)
        },
        "leakage_risks": leakage_notes,
    }

    output_json = reports_dir / "dataset_inspection.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nInspection results successfully exported to: {output_json}")
    print(f"Total time elapsed: {round(time.time() - start_time, 2)}s")

if __name__ == "__main__":
    main()
