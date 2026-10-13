#!/usr/bin/env python3
"""
GameRec Model Training Script (Phase 3).
Fits and persists Popularity and Content-Based models on the full processed games catalog.
"""

import sys
import time
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gamerec.data import get_default_paths, load_processed_catalog
from gamerec.popularity_model import PopularityRecommender
from gamerec.content_model import ContentBasedRecommender


def main():
    print("=" * 70)
    print("GameRec: Training Popularity & Content-Based Models")
    print("=" * 70)
    t0 = time.time()

    paths = get_default_paths()
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading processed games catalog...")
    catalog_df = load_processed_catalog(paths["processed_catalog"])
    print(f"Catalog shape: {catalog_df.shape} ({catalog_df['app_id'].n_unique():,} unique games)")

    # Model A: Popularity Baseline
    print("\n[2] Training Model A: Popularity Recommender (Bayesian Shrinkage)...")
    t_pop = time.time()
    pop_model = PopularityRecommender(support_quantile=0.75, default_min_reviews=50)
    pop_model.fit(catalog_df)
    pop_save_path = models_dir / "popularity_model.joblib"
    pop_model.save(pop_save_path)
    print(f"Popularity model trained and saved in {round(time.time() - t_pop, 2)}s to {pop_save_path}")

    # Model B: Content-Based Filtering
    print("\n[3] Training Model B: Content-Based Recommender (Multi-Modal TF-IDF)...")
    t_cnt = time.time()
    content_model = ContentBasedRecommender(
        weights={"tags": 0.60, "description": 0.25, "title": 0.15},
        max_desc_features=25000,
    )
    content_model.fit(catalog_df)
    content_save_path = models_dir / "content_model.joblib"
    content_model.save(content_save_path)
    print(f"Content model trained and saved in {round(time.time() - t_cnt, 2)}s to {content_save_path}")

    total_time = round(time.time() - t0, 2)
    print("\n" + "=" * 70)
    print(f"Phase 3 Model Fitting successfully completed in {total_time}s!")
    print(f"Artifacts saved in: {models_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
