"""
Unit and integration tests for Collaborative Filtering models.
"""

import pytest
import polars as pl
import numpy as np
from gamerec.collaborative_model import (
    temporal_train_test_split,
    build_interaction_matrix,
    ItemItemCollaborativeRecommender,
    MatrixFactorizationRecommender,
)


@pytest.fixture
def sample_interactions():
    """Generates a small multi-user interaction dataset for testing."""
    return pl.DataFrame({
        "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "app_id": [10, 20, 30, 40, 50, 10, 20, 30, 60, 70, 20, 30, 40, 80, 90],
        "review_id": list(range(1, 16)),
        "helpful": [0] * 15,
        "funny": [0] * 15,
        "date": [
            "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01",
            "2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15", "2020-05-15",
            "2020-01-20", "2020-02-20", "2020-03-20", "2020-04-20", "2020-05-20",
        ],
        "is_recommended": [True] * 15,
        "hours": [10.0, 25.0, 5.0, 50.0, 100.0, 15.0, 30.0, 8.0, 40.0, 120.0, 12.0, 22.0, 7.0, 35.0, 90.0],
        "feedback_type": [1] * 15,
        "implicit_confidence": [2.3] * 15,
    })


@pytest.fixture
def sample_catalog():
    return pl.DataFrame({
        "app_id": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "title": [
            "Game 10", "Game 20", "Game 30", "Game 40", "Game 50",
            "Game 60", "Game 70", "Game 80", "Game 90",
        ],
    })


def test_temporal_train_test_split(sample_interactions):
    """Verify that temporal splitting places the latest interaction in test."""
    train_df, test_df = temporal_train_test_split(sample_interactions, min_user_interactions=3)

    assert train_df.shape[0] == 12
    assert test_df.shape[0] == 3  # 1 per user

    # User 1's latest interaction is App 50 on 2020-05-01
    user1_test = test_df.filter(pl.col("user_id") == 1)
    assert user1_test["app_id"][0] == 50

    # User 1's train set must not contain App 50
    user1_train = train_df.filter(pl.col("user_id") == 1)
    assert 50 not in user1_train["app_id"].to_list()


def test_build_interaction_matrix(sample_interactions):
    """Verify matrix dimensions, non-zero values, and index mapping."""
    R, u2i, a2i, i2u, i2a = build_interaction_matrix(sample_interactions)

    assert R.shape == (3, 9)
    assert R.nnz == 15
    assert len(u2i) == 3
    assert len(a2i) == 9


def test_item_item_collaborative_model(sample_interactions, sample_catalog):
    """Verify Item-Item CF fitting, recommendations, and seen-item exclusion."""
    train_df, _ = temporal_train_test_split(sample_interactions, min_user_interactions=3)

    model = ItemItemCollaborativeRecommender(top_k_neighbors=5, shrinkage_lambda=1.0)
    model.fit(train_df, catalog_df=sample_catalog)

    assert model.is_fitted
    assert model.similarity_matrix is not None

    # Recommend for user 1 (played 10, 20, 30, 40 in train)
    recs = model.recommend_user(user_id=1, k=3)
    assert len(recs) > 0
    rec_ids = [r["app_id"] for r in recs]

    # Training items must be excluded
    for seen_id in [10, 20, 30, 40]:
        assert seen_id not in rec_ids

    # Candidate explanation must exist
    assert "explanation" in recs[0]


def test_matrix_factorization_model(sample_interactions, sample_catalog):
    """Verify Matrix Factorization (SVD) fitting and recommendations."""
    train_df, _ = temporal_train_test_split(sample_interactions, min_user_interactions=3)

    model = MatrixFactorizationRecommender(n_components=2, random_state=42)
    model.fit(train_df, catalog_df=sample_catalog)

    assert model.is_fitted
    assert model.item_factors is not None
    # Training set has 6 unique items (10, 20, 30, 40, 60, 80)
    assert model.item_factors.shape == (train_df["app_id"].n_unique(), 2)

    # Recommend for user 1
    recs = model.recommend_user(user_id=1, k=3)
    assert len(recs) > 0
    rec_ids = [r["app_id"] for r in recs]

    # Seen items must be excluded
    for seen_id in [10, 20, 30, 40]:
        assert seen_id not in rec_ids


def test_arbitrary_item_history_recommendations(sample_interactions, sample_catalog):
    """Verify real-time recommendations for arbitrary item lists (new user cold-start)."""
    train_df, _ = temporal_train_test_split(sample_interactions, min_user_interactions=3)

    item_model = ItemItemCollaborativeRecommender(top_k_neighbors=5)
    item_model.fit(train_df, catalog_df=sample_catalog)

    svd_model = MatrixFactorizationRecommender(n_components=2, random_state=42)
    svd_model.fit(train_df, catalog_df=sample_catalog)

    # User who played App 10 and 20
    recs_item = item_model.recommend_items([10, 20], k=2)
    recs_svd = svd_model.recommend_items([10, 20], k=2)

    assert len(recs_item) > 0
    assert len(recs_svd) > 0

    assert 10 not in [r["app_id"] for r in recs_item]
    assert 20 not in [r["app_id"] for r in recs_item]
    assert 10 not in [r["app_id"] for r in recs_svd]
    assert 20 not in [r["app_id"] for r in recs_svd]


def test_unknown_user_handling(sample_interactions, sample_catalog):
    """Verify that querying an unknown user gracefully returns empty without error."""
    train_df, _ = temporal_train_test_split(sample_interactions, min_user_interactions=3)

    model = ItemItemCollaborativeRecommender()
    model.fit(train_df, catalog_df=sample_catalog)

    recs = model.recommend_user(user_id=999999, k=5)
    assert recs == []


def test_collaborative_model_persistence(sample_interactions, sample_catalog, tmp_path):
    """Verify saving and reloading collaborative models."""
    train_df, _ = temporal_train_test_split(sample_interactions, min_user_interactions=3)

    item_model = ItemItemCollaborativeRecommender()
    item_model.fit(train_df, catalog_df=sample_catalog)
    path_item = tmp_path / "item_model.joblib"
    item_model.save(path_item)

    loaded_item = ItemItemCollaborativeRecommender.load(path_item)
    assert loaded_item.is_fitted
    assert len(loaded_item.recommend_user(user_id=1, k=2)) > 0
