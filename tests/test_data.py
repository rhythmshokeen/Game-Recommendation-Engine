"""
Unit and integration tests for data ingestion, cleaning, deduplication, and joins.
"""

import pytest
import polars as pl
from gamerec.data import get_default_paths, load_raw_games
from gamerec.preprocessing import (
    clean_games_catalog,
    clean_interactions,
    create_active_benchmark,
)


def test_default_paths():
    """Verify that default directory paths resolve properly."""
    paths = get_default_paths()
    assert "raw_dir" in paths
    assert "processed_dir" in paths
    assert "raw_games" in paths
    assert paths["raw_games"].exists()
    assert paths["raw_metadata"].exists()


def test_clean_games_catalog():
    """Verify schema downcasting, tag extraction, and Bayesian rating logic."""
    sample_games = pl.DataFrame({
        "app_id": [10, 20, 30],
        "title": [" Counter-Strike ", "Dota 2", "Half-Life"],
        "date_release": ["2012-08-21", "2013-07-09", "1998-11-19"],
        "win": [True, True, True],
        "mac": [True, True, False],
        "linux": [True, True, True],
        "rating": ["Very Positive", "Positive", "Overwhelmingly Positive"],
        "positive_ratio": [88, 82, 96],
        "user_reviews": [500000, 200000, 100000],
        "price_final": [14.99, 0.0, 9.99],
        "price_original": [14.99, 0.0, 9.99],
        "discount": [0.0, 0.0, 0.0],
        "steam_deck": [True, True, True],
    })

    sample_meta = pl.DataFrame({
        "app_id": [10, 20, 30],
        "description": ["Tactical FPS.", "", "Sci-fi shooter."],
        "tags": [["FPS", "Shooter", "Action"], [], ["Sci-Fi", "Classic"]],
    })

    catalog = clean_games_catalog(sample_games, sample_meta)

    assert catalog.shape[0] == 3
    assert catalog["app_id"].dtype == pl.UInt32
    assert catalog["title"][0] == "Counter-Strike"
    assert catalog["has_description"].to_list() == [True, False, True]
    assert catalog["has_tags"].to_list() == [True, False, True]
    assert catalog["is_free"].to_list() == [False, True, False]
    assert catalog["release_year"].to_list() == [2012, 2013, 1998]
    assert "bayesian_rating" in catalog.columns
    assert all(0.0 <= r <= 1.0 for r in catalog["bayesian_rating"].to_list())


def test_interaction_deduplication():
    """Verify that repeated (user_id, app_id) reviews resolve to the latest timestamp."""
    sample_rec = pl.LazyFrame({
        "app_id": [100, 100, 200],
        "user_id": [1, 1, 1],
        "review_id": [1001, 1002, 1003],
        "helpful": [0, 5, 1],
        "funny": [0, 0, 0],
        "date": ["2021-01-01", "2021-06-01", "2021-03-01"],
        "is_recommended": [True, False, True],  # Changed from True to False on later date
        "hours": [10.5, 45.0, 20.0],
    })

    clean_rec = clean_interactions(sample_rec, deduplicate=True).collect()

    assert clean_rec.shape[0] == 2
    # For user 1 and app 100, the latest record (2021-06-01, review_id 1002) must be preserved
    app100 = clean_rec.filter((pl.col("user_id") == 1) & (pl.col("app_id") == 100))
    assert app100.shape[0] == 1
    assert app100["is_recommended"][0] is False
    assert app100["hours"][0] == 45.0
    assert app100["feedback_type"][0] == -1


def test_feedback_semantics():
    """Verify explicit feedback encoding and confidence weights."""
    sample_rec = pl.LazyFrame({
        "app_id": [10, 20],
        "user_id": [1, 1],
        "review_id": [1, 2],
        "helpful": [0, 0],
        "funny": [0, 0],
        "date": ["2020-01-01", "2020-01-02"],
        "is_recommended": [True, False],
        "hours": [15.0, 2.0],
    })

    clean = clean_interactions(sample_rec, deduplicate=False).collect()

    # Explicit positive should be 1
    assert clean.filter(pl.col("app_id") == 10)["feedback_type"][0] == 1
    # Explicit negative should be -1
    assert clean.filter(pl.col("app_id") == 20)["feedback_type"][0] == -1
    # Implicit confidence should be ln(1 + hours)
    assert clean.filter(pl.col("app_id") == 10)["implicit_confidence"][0] > clean.filter(pl.col("app_id") == 20)["implicit_confidence"][0]


def test_benchmark_k_core_filtering():
    """Verify that create_active_benchmark properly filters active users and items."""
    # User 1 has 3 reviews, User 2 has 1 review
    # App 10 has 2 reviews, App 20 has 1 review, App 30 has 1 review
    sample_rec = pl.LazyFrame({
        "app_id": [10, 20, 30, 10],
        "user_id": [1, 1, 1, 2],
        "review_id": [1, 2, 3, 4],
        "helpful": [0, 0, 0, 0],
        "funny": [0, 0, 0, 0],
        "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        "is_recommended": [True, True, True, True],
        "hours": [10.0, 10.0, 10.0, 10.0],
    })

    # Benchmark with min_user_reviews=2, min_game_reviews=1
    bm = create_active_benchmark(
        sample_rec,
        min_user_reviews=2,
        min_game_reviews=1,
        max_users=None,
    )

    # Only user 1 should remain
    assert set(bm["user_id"].to_list()) == {1}
    assert bm.shape[0] == 3
