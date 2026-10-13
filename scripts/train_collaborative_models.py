#!/usr/bin/env python3
"""
GameRec Collaborative Models Training Script.
Performs leak-free temporal splitting on benchmark interactions, fits
Item-Item CF and Matrix Factorization models, and persists artifacts.
"""

import sys
import time
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gamerec.data import get_default_paths, load_processed_catalog, load_processed_interactions
from gamerec.collaborative_model import (
    temporal_train_test_split,
    ItemItemCollaborativeRecommender,
    MatrixFactorizationRecommender,
)


def main():
    print("=" * 70)
    print("GameRec: Training Collaborative Filtering Models")
    print("=" * 70)
    t0 = time.time()

    paths = get_default_paths()
    models_dir = project_root / "models"
    processed_dir = paths["processed_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading processed catalog and benchmark interactions...")
    catalog_df = load_processed_catalog(paths["processed_catalog"])
    benchmark_df = load_processed_interactions(paths["processed_benchmark"], lazy=False)
    print(f"Catalog shape: {catalog_df.shape}")
    print(f"Benchmark interactions: {benchmark_df.shape[0]:,} rows")

    print("\n[2] Performing leak-free temporal train/test split...")
    train_df, test_df = temporal_train_test_split(benchmark_df, min_user_interactions=5)

    # Persist split for rigorous offline evaluation
    train_path = processed_dir / "train_interactions.parquet"
    test_path = processed_dir / "test_interactions.parquet"
    train_df.write_parquet(train_path, compression="zstd")
    test_df.write_parquet(test_path, compression="zstd")
    print(f"Saved Train interactions to {train_path} ({round(train_path.stat().st_size / (1024*1024), 2)} MB)")
    print(f"Saved Test interactions to  {test_path} ({round(test_path.stat().st_size / (1024*1024), 2)} MB)")

    # Model 1: Item-Item Collaborative Filtering
    print("\n[3] Training Item-Item Collaborative Recommender (Bayesian Shrinkage lambda=15.0, Top-100)...")
    t_item = time.time()
    item_model = ItemItemCollaborativeRecommender(
        top_k_neighbors=100,
        shrinkage_lambda=15.0,
        alpha_hours=0.5,
    )
    item_model.fit(train_df, catalog_df=catalog_df)
    item_save_path = models_dir / "item_collaborative_model.joblib"
    item_model.save(item_save_path)
    print(f"Item-Item CF model trained and saved in {round(time.time() - t_item, 2)}s to {item_save_path}")

    # Model 2: Matrix Factorization (TruncatedSVD)
    print("\n[4] Training Matrix Factorization Recommender (64 Latent Factors, random_state=42)...")
    t_svd = time.time()
    svd_model = MatrixFactorizationRecommender(
        n_components=64,
        random_state=42,
        alpha_hours=0.5,
    )
    svd_model.fit(train_df, catalog_df=catalog_df)
    svd_save_path = models_dir / "matrix_factorization_model.joblib"
    svd_model.save(svd_save_path)
    print(f"Matrix Factorization model trained and saved in {round(time.time() - t_svd, 2)}s to {svd_save_path}")

    total_time = round(time.time() - t0, 2)
    print("\n" + "=" * 70)
    print(f"Collaborative Model Training successfully completed in {total_time}s!")
    print(f"Artifacts saved in: {models_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
