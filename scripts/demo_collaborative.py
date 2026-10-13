#!/usr/bin/env python3
"""
GameRec Collaborative Filtering Demonstration & Comparison Script.
Compares Item-Item Collaborative Filtering (Neighborhood) with
Matrix Factorization (TruncatedSVD) on actual users from the benchmark dataset.
Inspects recommendations, explanations, seen-item exclusions, and failure modes.
"""

import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import polars as pl
from gamerec.collaborative_model import (
    ItemItemCollaborativeRecommender,
    MatrixFactorizationRecommender,
)


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text.upper()}")
    print("=" * 80)


def main():
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"

    print("Loading models and catalog...")
    item_model = ItemItemCollaborativeRecommender.load(models_dir / "item_collaborative_model.joblib")
    svd_model = MatrixFactorizationRecommender.load(models_dir / "matrix_factorization_model.joblib")

    train_df = pl.read_parquet(processed_dir / "train_interactions.parquet")
    test_df = pl.read_parquet(processed_dir / "test_interactions.parquet")
    catalog_df = pl.read_parquet(processed_dir / "games_catalog.parquet")
    cat_map = dict(zip(catalog_df["app_id"].to_list(), catalog_df["title"].to_list()))

    eval_users = [
        (4002, "Multiplayer / Survival & Action Gamer"),
        (550, "Action RPG / Looter Shooter Gamer"),
        (535, "Co-op / Sandbox & Indie Gamer"),
    ]

    for uid, persona in eval_users:
        print_banner(f"User {uid}: {persona}")
        
        # Display training history
        u_train = train_df.filter(pl.col("user_id") == uid).sort("hours", descending=True)
        u_test = test_df.filter(pl.col("user_id") == uid)

        print(f"Known Training History ({len(u_train)} games):")
        for r in u_train.head(5).iter_rows(named=True):
            print(f"  - {cat_map.get(r['app_id'], r['app_id'])} ({r['hours']:.1f} hrs, Recommended: {r['is_recommended']})")

        print("\nHeld-out Test Interaction (Target for Evaluation):")
        for r in u_test.iter_rows(named=True):
            print(f"  * {cat_map.get(r['app_id'], r['app_id'])} ({r['hours']:.1f} hrs, Date: {r['date']})")

        # 1. Item-Item CF Recommendations
        print("\n--- Model 1: Item-Item Collaborative Filtering (Neighborhood) ---")
        item_recs = item_model.recommend_user(user_id=uid, k=5)
        for r in item_recs:
            print(f"  [{r['rank']}] {r['title']} (App ID: {r['app_id']}) | Score: {r['score']:.4f}")
            print(f"      {r['explanation']}")

        # 2. Matrix Factorization (SVD) Recommendations
        print("\n--- Model 2: Matrix Factorization (64 Latent Factors SVD) ---")
        svd_recs = svd_model.recommend_user(user_id=uid, k=5)
        for r in svd_recs:
            print(f"  [{r['rank']}] {r['title']} (App ID: {r['app_id']}) | Latent Score: {r['score']:.4f}")
            print(f"      {r['explanation']}")

        # Check overlap between models
        item_ids = {r["app_id"] for r in item_recs}
        svd_ids = {r["app_id"] for r in svd_recs}
        overlap = item_ids.intersection(svd_ids)
        overlap_titles = [cat_map.get(aid, aid) for aid in overlap]
        print(f"\nModel Agreement: {len(overlap)}/5 games overlap between Item-Item and SVD: {overlap_titles}")


if __name__ == "__main__":
    main()
