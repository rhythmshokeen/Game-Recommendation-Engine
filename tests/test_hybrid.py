"""
Automated unit and integration tests for Hybrid Recommendation Engine and Ranking.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from gamerec.content_model import ContentBasedRecommender
from gamerec.collaborative_model import ItemItemCollaborativeRecommender
from gamerec.popularity_model import PopularityRecommender
from gamerec.ranking import min_max_scale, combine_hybrid_scores, maximal_marginal_relevance
from gamerec.hybrid_model import HybridRecommender


@pytest.fixture
def mock_dataset():
    """Builds a consistent synthetic catalog and interaction dataset."""
    catalog = pl.DataFrame({
        "app_id": [10, 20, 30, 40, 50, 60],
        "title": [
            "Hades",
            "Dead Cells",
            "Slay the Spire",
            "Counter-Strike",
            "Left 4 Dead 2",
            "Indie Mystery",
        ],
        "date_release": ["2020-09-17"] * 6,
        "win": [True] * 6,
        "mac": [True] * 6,
        "linux": [True] * 6,
        "rating": ["Overwhelmingly Positive"] * 6,
        "positive_ratio": [98, 97, 97, 88, 97, 85],
        "user_reviews": [200000, 120000, 110000, 7000000, 500000, 500],
        "price_final": [24.99, 24.99, 24.99, 14.99, 9.99, 4.99],
        "price_original": [24.99, 24.99, 24.99, 14.99, 9.99, 4.99],
        "discount": [0.0] * 6,
        "steam_deck": [True] * 6,
        "description": [
            "Hades rogue-like dungeon crawler with god powers.",
            "Dead Cells action rogue-like platformer.",
            "Slay the Spire deckbuilding card rogue-like.",
            "Tactical first-person shooter with bomb defusal.",
            "Cooperative zombie shooter with survivors.",
            "Indie adventure game.",
        ],
        "tags": [
            ["Roguelike", "Action", "Indie"],
            ["Roguelike", "Action", "Metroidvania"],
            ["Roguelike", "Strategy", "Indie"],
            ["FPS", "Shooter", "Action"],
            ["FPS", "Zombies", "Co-op"],
            ["Indie", "Adventure"],
        ],
        "has_description": [True] * 6,
        "has_tags": [True] * 6,
        "release_year": [2020] * 6,
        "is_free": [False] * 6,
        "tags_str": [
            "Roguelike Action Indie",
            "Roguelike Action Metroidvania",
            "Roguelike Strategy Indie",
            "FPS Shooter Action",
            "FPS Zombies Co-op",
            "Indie Adventure",
        ],
        "bayesian_rating": [0.98, 0.97, 0.97, 0.88, 0.97, 0.84],
    })

    interactions = pl.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 2, 3, 3],
        "app_id": [10, 20, 30, 40, 10, 20, 50, 40, 50],
        "review_id": list(range(1, 10)),
        "helpful": [0] * 9,
        "funny": [0] * 9,
        "date": ["2020-01-01"] * 9,
        "is_recommended": [True] * 9,
        "hours": [15.0] * 9,
        "feedback_type": [1] * 9,
        "implicit_confidence": [2.7] * 9,
    })

    return catalog, interactions


@pytest.fixture
def fitted_hybrid_engine(mock_dataset):
    catalog, interactions = mock_dataset

    pop = PopularityRecommender(support_quantile=0.5, default_min_reviews=10)
    pop.fit(catalog)

    content = ContentBasedRecommender()
    content.fit(catalog)

    collab = ItemItemCollaborativeRecommender(top_k_neighbors=5, shrinkage_lambda=1.0)
    collab.fit(interactions, catalog_df=catalog)

    hybrid = HybridRecommender(
        content_model=content,
        collaborative_model=collab,
        popularity_model=pop,
        default_alpha=0.5,
        min_history_for_hybrid=2,
    )
    return hybrid


def test_ranking_min_max_scale():
    """Verify min_max_scale normalizes arrays properly and handles constants."""
    scores = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    scaled = min_max_scale(scores)
    assert np.isclose(scaled[0], 0.0)
    assert np.isclose(scaled[2], 1.0)
    assert np.isclose(scaled[1], 0.5)

    # Constant array
    const = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    scaled_const = min_max_scale(const)
    assert all(scaled_const == 1.0)


def test_combine_hybrid_scores_alpha_boundary():
    """Verify combine_hybrid_scores for alpha=1.0 and alpha=0.0."""
    candidates = [10, 20]
    content_scores = {10: 0.9, 20: 0.4}
    collab_scores = {10: 0.2, 20: 0.8}

    # Alpha = 1.0 (Content only)
    c_only = combine_hybrid_scores(candidates, content_scores, collab_scores, alpha=1.0)
    assert c_only[10]["hybrid_score"] == 0.9
    assert c_only[20]["hybrid_score"] == 0.4

    # Alpha = 0.0 (Collaborative only)
    cf_only = combine_hybrid_scores(candidates, content_scores, collab_scores, alpha=0.0)
    assert cf_only[10]["hybrid_score"] == 0.2
    assert cf_only[20]["hybrid_score"] == 0.8


def test_combine_hybrid_scores_intermediate():
    """Verify intermediate alpha weighting."""
    candidates = [10]
    content_scores = {10: 0.8}
    collab_scores = {10: 0.4}

    # Alpha = 0.5: 0.5 * 0.8 + 0.5 * 0.4 = 0.6
    res = combine_hybrid_scores(candidates, content_scores, collab_scores, alpha=0.5)
    assert np.isclose(res[10]["hybrid_score"], 0.6)

    # Alpha = 0.25: 0.25 * 0.8 + 0.75 * 0.4 = 0.5
    res25 = combine_hybrid_scores(candidates, content_scores, collab_scores, alpha=0.25)
    assert np.isclose(res25[10]["hybrid_score"], 0.5)


def test_missing_collaborative_score():
    """Verify handling when an item has content score but no collaborative score."""
    candidates = [10, 999]  # 999 is cold-start item
    content_scores = {10: 0.8, 999: 0.6}
    collab_scores = {10: 0.5}  # 999 missing

    res = combine_hybrid_scores(candidates, content_scores, collab_scores, alpha=0.5)
    # 999 receives alpha * content_score = 0.5 * 0.6 = 0.3
    assert np.isclose(res[999]["hybrid_score"], 0.3)


def test_hybrid_recommender_warm_user(fitted_hybrid_engine):
    """Verify recommendations for warm user (user 2 has 3 interactions)."""
    # User 2 played 10, 20, 50
    recs = fitted_hybrid_engine.recommend_user(user_id=2, k=2, alpha=0.5)
    assert len(recs) == 2
    rec_ids = [r["app_id"] for r in recs]

    # Seen items must be excluded
    for seen_id in [10, 20, 50]:
        assert seen_id not in rec_ids

    # Structure checks
    assert "hybrid_score" in recs[0]
    assert "explanation" in recs[0]
    assert recs[0]["alpha"] == 0.5


def test_hybrid_recommender_alpha_one_and_zero(fitted_hybrid_engine):
    """Verify alpha=1.0 and alpha=0.0 in HybridRecommender."""
    # Alpha = 1.0 (Content only)
    recs_c = fitted_hybrid_engine.recommend_user(user_id=2, k=2, alpha=1.0)
    assert recs_c[0]["alpha"] == 1.0
    assert "Content-only" in recs_c[0]["explanation"]

    # Alpha = 0.0 (Collaborative only)
    recs_cf = fitted_hybrid_engine.recommend_user(user_id=2, k=2, alpha=0.0)
    assert recs_cf[0]["alpha"] == 0.0
    assert "Collaborative-only" in recs_cf[0]["explanation"]


def test_hybrid_recommender_cold_start_user(fitted_hybrid_engine):
    """Verify that a user with 0 interactions routes to Popularity Baseline."""
    recs = fitted_hybrid_engine.recommend_user(user_id=9999, k=3)
    assert len(recs) == 3
    # Top popular items
    assert "Community baseline" in recs[0]["explanation"]


def test_hybrid_recommender_deterministic(fitted_hybrid_engine):
    """Verify rankings are strictly deterministic for fixed inputs."""
    recs1 = fitted_hybrid_engine.recommend_user(user_id=2, k=3, alpha=0.5)
    recs2 = fitted_hybrid_engine.recommend_user(user_id=2, k=3, alpha=0.5)

    assert [r["app_id"] for r in recs1] == [r["app_id"] for r in recs2]
    assert [r["score"] for r in recs1] == [r["score"] for r in recs2]


def test_hybrid_recommender_seen_item_exclusion(fitted_hybrid_engine):
    """Verify external exclude_seen items are respected."""
    recs = fitted_hybrid_engine.recommend_user(user_id=2, k=3, exclude_seen={30})
    rec_ids = [r["app_id"] for r in recs]
    assert 30 not in rec_ids


def test_diversity_maximal_marginal_relevance():
    """Verify MMR diversity reranking with franchise limits."""
    candidates = [
        {"app_id": 1, "title": "Tomb Raider I", "score": 0.95, "tags": ["Action", "Adventure"]},
        {"app_id": 2, "title": "Tomb Raider II", "score": 0.94, "tags": ["Action", "Adventure"]},
        {"app_id": 3, "title": "Tomb Raider III", "score": 0.93, "tags": ["Action", "Adventure"]},
        {"app_id": 4, "title": "Unrelated RPG", "score": 0.85, "tags": ["RPG", "Fantasy"]},
    ]

    # Max 2 per franchise -> Tomb Raider III should be capped out
    diverse = maximal_marginal_relevance(
        candidates=candidates,
        top_k=3,
        max_per_franchise=2,
        diversity_lambda=0.5,
    )

    diverse_ids = [c["app_id"] for c in diverse]
    assert len(diverse) == 3
    # App 4 must be selected because only 2 Tomb Raider games are allowed
    assert 4 in diverse_ids
