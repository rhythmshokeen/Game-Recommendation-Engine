#!/usr/bin/env python3
"""
GameRec: Models Demo Script (Popularity Baseline & Content-Based Filtering).
Demonstrates real recommendations on the Steam dataset.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gamerec.popularity_model import PopularityRecommender
from gamerec.content_model import ContentBasedRecommender


def main():
    models_dir = project_root / "models"
    pop_model = PopularityRecommender.load(models_dir / "popularity_model.joblib")
    content_model = ContentBasedRecommender.load(models_dir / "content_model.joblib")

    print("\n--- 1. Popularity Baseline Top 5 ---")
    for r in pop_model.recommend(k=5, ranking_metric="weighted_popularity"):
        print(f"#{r['rank']}: {r['title']} (Score: {r['score']:.2f}, {r['positive_ratio']}% on {r['user_reviews']:,} reviews)")

    print("\n--- 2. Content-Based Top 3 for Tomb Raider (203160) ---")
    for r in content_model.recommend_similar(app_id=203160, k=3):
        print(f"#{r['rank']}: {r['title']} (Sim: {r['similarity_score']:.3f}) | Shared: {r['shared_tags'][:4]}")

    print("\n--- 3. Content-Based Top 3 for Mount & Blade (48700) ---")
    for r in content_model.recommend_similar(app_id=48700, k=3):
        print(f"#{r['rank']}: {r['title']} (Sim: {r['similarity_score']:.3f}) | Shared: {r['shared_tags'][:4]}")


if __name__ == "__main__":
    main()
