"""
Automated unit and regression tests for Popularity and Content-Based models.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from gamerec.popularity_model import PopularityRecommender
from gamerec.content_model import ContentBasedRecommender


@pytest.fixture
def sample_catalog():
    """Generates a representative small catalog DataFrame for unit testing."""
    return pl.DataFrame({
        "app_id": [10, 20, 30, 40, 50],
        "title": [
            "Hades",
            "Dead Cells",
            "Slay the Spire",
            "Counter-Strike",
            "Unknown Mystery Game",
        ],
        "date_release": ["2020-09-17", "2018-08-07", "2019-01-23", "2012-08-21", "2021-01-01"],
        "win": [True, True, True, True, False],
        "mac": [True, True, True, True, False],
        "linux": [False, True, True, True, False],
        "rating": ["Overwhelmingly Positive", "Overwhelmingly Positive", "Overwhelmingly Positive", "Very Positive", "Mixed"],
        "positive_ratio": [98, 97, 97, 88, 50],
        "user_reviews": [200000, 120000, 110000, 7000000, 20],
        "price_final": [24.99, 24.99, 24.99, 14.99, 4.99],
        "price_original": [24.99, 24.99, 24.99, 14.99, 4.99],
        "discount": [0.0, 0.0, 0.0, 0.0, 0.0],
        "steam_deck": [True, True, True, True, False],
        "description": [
            "Defy the god of the dead as you hack and slash out of the Underworld in this rogue-like dungeon crawler.",
            "Dead Cells is a rogue-like action-platformer with challenging combat and procedurally generated castles.",
            "A deckbuilding rogue-like adventure where you craft a unique card deck and slay monsters.",
            "Tactical first-person team-based shooter with defusal and hostage rescue missions.",
            "",  # Empty description test case
        ],
        "tags": [
            ["Roguelike", "Action", "Indie", "Great Soundtrack", "Singleplayer"],
            ["Roguelike", "Action", "Metroidvania", "Pixel Graphics", "Difficult"],
            ["Roguelike", "Deckbuilding", "Card Game", "Strategy", "Indie"],
            ["FPS", "Shooter", "Multiplayer", "Action", "Competitive"],
            [],  # Empty tags test case
        ],
        "has_description": [True, True, True, True, False],
        "has_tags": [True, True, True, True, False],
        "release_year": [2020, 2018, 2019, 2012, 2021],
        "is_free": [False, False, False, False, False],
        "tags_str": [
            "Roguelike Action Indie Great_Soundtrack Singleplayer",
            "Roguelike Action Metroidvania Pixel_Graphics Difficult",
            "Roguelike Deckbuilding Card_Game Strategy Indie",
            "FPS Shooter Multiplayer Action Competitive",
            "",
        ],
        "bayesian_rating": [0.978, 0.968, 0.968, 0.880, 0.550],
    })


# --- Popularity Model Tests ---

def test_popularity_recommender_fit(sample_catalog):
    """Verify fitting calculates priors and shrinkage scores."""
    model = PopularityRecommender(support_quantile=0.5, default_min_reviews=10)
    model.fit(sample_catalog)

    assert model.is_fitted
    assert model.prior_mean_positive_ratio > 0.0
    assert model.regularization_m > 0.0
    assert "computed_bayesian_rating" in model.catalog_df.columns
    assert "weighted_pop_score" in model.catalog_df.columns


def test_popularity_recommender_overall(sample_catalog):
    """Verify overall ranking order and top recommendations."""
    model = PopularityRecommender(support_quantile=0.5, default_min_reviews=10)
    model.fit(sample_catalog)

    recs = model.recommend(k=3, ranking_metric="weighted_popularity")
    assert len(recs) == 3
    # Counter-Strike has 7M reviews, should dominate weighted_popularity
    assert recs[0]["app_id"] == 40
    assert recs[0]["rank"] == 1
    assert "explanation" in recs[0]

    # Test Bayesian quality metric (where Hades should be top)
    bayes_recs = model.recommend(k=3, ranking_metric="bayesian_rating")
    assert bayes_recs[0]["app_id"] == 10  # Hades (98% on 200k reviews)


def test_popularity_recommender_tag_filter(sample_catalog):
    """Verify tag-specific popularity filtering."""
    model = PopularityRecommender(default_min_reviews=10)
    model.fit(sample_catalog)

    # Filter by Roguelike (Counter-Strike must be excluded)
    rogue_recs = model.recommend(k=5, tag="Roguelike")
    assert len(rogue_recs) == 3
    app_ids = [r["app_id"] for r in rogue_recs]
    assert 40 not in app_ids
    assert set(app_ids) == {10, 20, 30}


def test_popularity_recommender_exclusion(sample_catalog):
    """Verify exclusion of seen games."""
    model = PopularityRecommender(default_min_reviews=10)
    model.fit(sample_catalog)

    recs = model.recommend(k=2, exclude_app_ids={40})
    app_ids = [r["app_id"] for r in recs]
    assert 40 not in app_ids


def test_popularity_recommender_platform_filter(sample_catalog):
    """Verify platform availability filtering."""
    model = PopularityRecommender(default_min_reviews=10)
    model.fit(sample_catalog)

    # Linux filter (Hades has linux=False in fixture, others have True)
    linux_recs = model.recommend(k=5, platform="linux")
    linux_ids = [r["app_id"] for r in linux_recs]
    assert 10 not in linux_ids


# --- Content-Based Model Tests ---

def test_content_recommender_fit(sample_catalog):
    """Verify TF-IDF fitting and sparse matrix generation."""
    model = ContentBasedRecommender()
    model.fit(sample_catalog)

    assert model.is_fitted
    assert model.feature_matrix is not None
    assert model.feature_matrix.shape[0] == 5
    assert len(model.app_id_to_idx) == 5


def test_content_recommender_similarity(sample_catalog):
    """Verify similar game recommendations and thematic coherence."""
    model = ContentBasedRecommender()
    model.fit(sample_catalog)

    # Recommend games similar to Hades (app_id 10)
    recs = model.recommend_similar(app_id=10, k=2)

    assert len(recs) == 2
    # Self should never be recommended
    rec_ids = [r["app_id"] for r in recs]
    assert 10 not in rec_ids

    # Dead Cells (app_id 20) and Slay the Spire (app_id 30) share Roguelike, Action, Indie
    assert rec_ids[0] in [20, 30]
    assert len(recs[0]["shared_tags"]) > 0
    assert "Roguelike" in recs[0]["shared_tags"]


def test_content_recommender_user_profile(sample_catalog):
    """Verify multi-item user preference profile recommendations."""
    model = ContentBasedRecommender()
    model.fit(sample_catalog)

    # User likes Dead Cells (20) and Slay the Spire (30)
    recs = model.recommend_user_profile(
        liked_games=[{"app_id": 20, "weight": 2.0}, {"app_id": 30, "weight": 1.0}],
        k=2,
    )

    rec_ids = [r["app_id"] for r in recs]
    # Liked games must be excluded
    assert 20 not in rec_ids
    assert 30 not in rec_ids
    # Hades (10) shares Roguelike themes with both
    assert rec_ids[0] == 10
    assert "explanation" in recs[0]


def test_content_recommender_invalid_id(sample_catalog):
    """Verify error handling on non-existent app_id."""
    model = ContentBasedRecommender()
    model.fit(sample_catalog)

    with pytest.raises(ValueError, match="not found in catalog"):
        model.recommend_similar(app_id=999999)


def test_content_recommender_empty_metadata(sample_catalog):
    """Verify handling of games with empty descriptions and empty tags (app_id 50)."""
    model = ContentBasedRecommender()
    model.fit(sample_catalog)

    # Querying game with empty metadata should not crash
    recs = model.recommend_similar(app_id=50, k=2)
    assert isinstance(recs, list)


def test_content_recommender_no_duplicate_results(sample_catalog):
    """Ensure returned recommendation list contains strictly unique app_ids."""
    model = ContentBasedRecommender()
    model.fit(sample_catalog)

    recs = model.recommend_similar(app_id=10, k=4)
    rec_ids = [r["app_id"] for r in recs]
    assert len(rec_ids) == len(set(rec_ids))


def test_model_serialization(sample_catalog, tmp_path):
    """Verify model saving and reloading."""
    pop_model = PopularityRecommender()
    pop_model.fit(sample_catalog)
    pop_path = tmp_path / "popularity.joblib"
    pop_model.save(pop_path)

    loaded_pop = PopularityRecommender.load(pop_path)
    assert loaded_pop.is_fitted
    assert len(loaded_pop.recommend(k=2)) == 2

    content_model = ContentBasedRecommender()
    content_model.fit(sample_catalog)
    content_path = tmp_path / "content.joblib"
    content_model.save(content_path)

    loaded_content = ContentBasedRecommender.load(content_path)
    assert loaded_content.is_fitted
    assert len(loaded_content.recommend_similar(app_id=10, k=2)) == 2
