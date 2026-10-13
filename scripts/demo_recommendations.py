#!/usr/bin/env python3
"""
GameRec Recommendation Demonstration Script.
Evaluates Popularity and Content-Based models on real games from the Steam catalog.
Inspects returned metadata, shared features, and qualitative relevance.
"""

import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gamerec.popularity_model import PopularityRecommender
from gamerec.content_model import ContentBasedRecommender


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def main():
    models_dir = project_root / "models"
    pop_model_path = models_dir / "popularity_model.joblib"
    content_model_path = models_dir / "content_model.joblib"

    print("Loading models from artifacts...")
    pop_model = PopularityRecommender.load(pop_model_path)
    content_model = ContentBasedRecommender.load(content_model_path)

    # -------------------------------------------------------------
    # DEMO 1: Popularity Baseline (Overall & Tag-Specific)
    # -------------------------------------------------------------
    print_banner("1. Popularity Baseline: Overall Top 5 (Weighted Popularity)")
    overall_popular = pop_model.recommend(k=5, ranking_metric="weighted_popularity")
    for item in overall_popular:
        print(f"[{item['rank']}] {item['title']} (App ID: {item['app_id']})")
        print(f"    Score: {item['score']:.2f} | Positive: {item['positive_ratio']}% | Reviews: {item['user_reviews']:,}")
        print(f"    Explanation: {item['explanation']}\n")

    print_banner("2. Popularity Baseline: Top 5 in Genre 'Roguelike'")
    rogue_popular = pop_model.recommend(k=5, tag="Roguelike", ranking_metric="bayesian_rating", min_reviews=500)
    for item in rogue_popular:
        print(f"[{item['rank']}] {item['title']} (App ID: {item['app_id']})")
        print(f"    Bayesian Quality Score: {item['bayesian_rating']:.3f} | Positive: {item['positive_ratio']}% | Reviews: {item['user_reviews']:,}")
        print(f"    Tags: {item['tags'][:5]}")
        print(f"    Explanation: {item['explanation']}\n")

    # -------------------------------------------------------------
    # DEMO 2: Content-Based Filtering: Single Seed Game
    # -------------------------------------------------------------
    test_games = [
        (220, "Half-Life 2", "FPS / Sci-Fi / Classic"),
        (203160, "Tomb Raider", "Action-Adventure / Female Protagonist"),
        (48700, "Mount & Blade: Warband", "Medieval / Sandbox RPG / Strategy"),
        (698780, "Doki Doki Literature Club!", "Psychological Horror / Visual Novel"),
    ]

    for app_id, title, genre_theme in test_games:
        print_banner(f"3. Content-Based Recommendations for '{title}' (Theme: {genre_theme})")
        recs = content_model.recommend_similar(app_id=app_id, k=5)
        for item in recs:
            shared = ", ".join(item["shared_tags"][:6]) if item["shared_tags"] else "None (Title/Desc matched)"
            print(f"[{item['rank']}] {item['title']} (App ID: {item['app_id']}) | Cosine Sim: {item['similarity_score']:.3f}")
            print(f"    Shared Community Tags: [{shared}]")
            print(f"    Positive Ratio: {item['positive_ratio']}% | Reviews: {item['user_reviews']:,} | Price: ${item['price']:.2f}")
            print(f"    Explanation: {item['explanation']}\n")

    # -------------------------------------------------------------
    # DEMO 3: Multi-Game User Taste Profile
    # -------------------------------------------------------------
    print_banner("4. Content-Based Recommendations for Multi-Item User Taste Profile")
    print("User Liked Games Profile:")
    print(" - 'Mount & Blade: Warband' (App ID 48700) [Weight: 2.0 - Heavy Playtime / Favorite]")
    print(" - 'Tomb Raider' (App ID 203160)           [Weight: 1.0 - Completed]")

    user_profile = [
        {"app_id": 48700, "weight": 2.0},
        {"app_id": 203160, "weight": 1.0},
    ]

    profile_recs = content_model.recommend_user_profile(user_profile, k=5, min_reviews=100)
    for item in profile_recs:
        shared = ", ".join(item["shared_tags"][:6]) if item["shared_tags"] else "None"
        print(f"[{item['rank']}] {item['title']} (App ID: {item['app_id']}) | Profile Sim: {item['similarity_score']:.3f}")
        print(f"    Overlapping Tags: [{shared}]")
        print(f"    Explanation: {item['explanation']}\n")


if __name__ == "__main__":
    main()
