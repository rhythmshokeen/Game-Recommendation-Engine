#!/usr/bin/env python3
"""
GameRec Hybrid Engine Demonstration Script.
Demonstrates the Hybrid Recommender across alpha spectrum (0.0 to 1.0),
cold-start routing, diversity reranking, and truthful explanation generation.
"""

import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import polars as pl
from gamerec.content_model import ContentBasedRecommender
from gamerec.collaborative_model import ItemItemCollaborativeRecommender
from gamerec.popularity_model import PopularityRecommender
from gamerec.hybrid_model import HybridRecommender


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text.upper()}")
    print("=" * 80)


def main():
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"

    print("Loading component models from artifacts...")
    content_model = ContentBasedRecommender.load(models_dir / "content_model.joblib")
    collab_model = ItemItemCollaborativeRecommender.load(models_dir / "item_collaborative_model.joblib")
    pop_model = PopularityRecommender.load(models_dir / "popularity_model.joblib")

    # Instantiate Hybrid Engine
    hybrid_engine = HybridRecommender(
        content_model=content_model,
        collaborative_model=collab_model,
        popularity_model=pop_model,
        default_alpha=0.5,
        min_history_for_hybrid=3,
        diversity_lambda=0.7,
        max_per_franchise=2,
    )

    # Save configuration artifact for reproducible inference
    config_path = models_dir / "hybrid_config.json"
    hybrid_engine.save_config(config_path)

    catalog_df = pl.read_parquet(processed_dir / "games_catalog.parquet")
    train_df = pl.read_parquet(processed_dir / "train_interactions.parquet")
    cat_map = dict(zip(catalog_df["app_id"].to_list(), catalog_df["title"].to_list()))

    # -------------------------------------------------------------
    # DEMO 1: Warm User — Alpha Spectrum Comparison (User 550)
    # -------------------------------------------------------------
    uid = 550
    u_train = train_df.filter(pl.col("user_id") == uid).sort("hours", descending=True)
    print_banner(f"1. Warm User {uid} — Alpha Parameter Sensitivity")
    print(f"User Training History ({len(u_train)} games):")
    for r in u_train.head(4).iter_rows(named=True):
        print(f"  - {cat_map.get(r['app_id'], r['app_id'])} ({r['hours']:.1f} hrs, Recommended: {r['is_recommended']})")

    for alpha in [1.0, 0.7, 0.5, 0.3, 0.0]:
        mode_desc = (
            "Content-Only" if alpha == 1.0 else
            "Collab-Only" if alpha == 0.0 else
            f"Hybrid (α={alpha:.1f})"
        )
        print(f"\n>>> Alpha = {alpha:.1f} ({mode_desc}) Recommendations:")
        recs = hybrid_engine.recommend_user(user_id=uid, k=3, alpha=alpha, use_diversity=True)
        for r in recs:
            print(f"  #{r['rank']}: {r['title']} (App ID: {r['app_id']})")
            print(f"      Score: {r['score']:.4f} [Content: {r['content_score']:.3f}, Collab: {r['collab_score']:.3f}]")
            print(f"      Explanation: {r['explanation']}")

    # -------------------------------------------------------------
    # DEMO 2: Cold-Start Handling
    # -------------------------------------------------------------
    print_banner("2. Cold-Start Handling Demonstrations")

    # A. Completely New User (0 History)
    print(">>> Scenario A: Completely New User (0 Reviews):")
    new_user_recs = hybrid_engine.recommend_user(user_id=99999999, k=3)
    for r in new_user_recs:
        print(f"  #{r['rank']}: {r['title']} (Score: {r['score']:.2f})")
        print(f"      Explanation: {r['explanation']}")

    # B. Interactive Guest Session (Selected 2 Preferences)
    print("\n>>> Scenario B: Interactive Guest Session (Liked: Half-Life 2 & Portal 2):")
    guest_recs = hybrid_engine.recommend_interactive(
        seed_games=[{"app_id": 220, "weight": 2.0}, {"app_id": 620, "weight": 1.5}],
        k=3,
        alpha=0.5,
    )
    for r in guest_recs:
        print(f"  #{r['rank']}: {r['title']} (Score: {r['score']:.4f})")
        print(f"      [Content: {r['content_score']:.3f}, Collab: {r['collab_score']:.3f}]")
        print(f"      Explanation: {r['explanation']}")

    # -------------------------------------------------------------
    # DEMO 3: Diversity & Redundancy Suppression (MMR)
    # -------------------------------------------------------------
    print_banner("3. Diversity Control: Maximal Marginal Relevance (MMR)")
    print("Testing recommendations for 'Tomb Raider' fans with vs. without diversity controls:")

    # Without diversity (lambda=1.0, max_franchise=0)
    hybrid_engine.diversity_lambda = 1.0
    hybrid_engine.max_per_franchise = 0
    recs_no_div = hybrid_engine.recommend_interactive(seed_games=[203160], k=5, alpha=0.5)
    print("Without Diversity Control (Standard Greedy Rank):")
    for r in recs_no_div:
        print(f"  #{r['rank']}: {r['title']}")

    # With diversity (lambda=0.7, max_franchise=2)
    hybrid_engine.diversity_lambda = 0.7
    hybrid_engine.max_per_franchise = 2
    recs_with_div = hybrid_engine.recommend_interactive(seed_games=[203160], k=5, alpha=0.5)
    print("\nWith Diversity Control (MMR λ=0.7, Max Franchise=2):")
    for r in recs_with_div:
        print(f"  #{r['rank']}: {r['title']}")


if __name__ == "__main__":
    main()
