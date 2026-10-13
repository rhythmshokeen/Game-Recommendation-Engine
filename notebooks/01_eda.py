#!/usr/bin/env python3
"""
GameRec: Exploratory Data Analysis (EDA) Interactive Script.
Accompanies notebooks/01_eda.ipynb for headless execution and quick profiling.
"""

import sys
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Setup path to src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gamerec.data import load_processed_catalog, load_processed_interactions, get_default_paths

def main():
    paths = get_default_paths(project_root)
    print("Project paths initialized:", paths["project_root"])

    print("\n--- 1. Games Catalog Inspection ---")
    catalog_df = load_processed_catalog(paths["processed_catalog"])
    print(f"Catalog Shape: {catalog_df.shape}")
    print(catalog_df.select(["app_id", "title", "positive_ratio", "user_reviews", "price_final", "bayesian_rating", "has_tags"]).head(5))

    print("\n--- 2. Active Benchmark Interactions Inspection ---")
    benchmark_df = load_processed_interactions(paths["processed_benchmark"], lazy=False)
    print(f"Benchmark Interactions Shape: {benchmark_df.shape}")
    print(f"Unique Users: {benchmark_df['user_id'].n_unique():,}")
    print(f"Unique Games: {benchmark_df['app_id'].n_unique():,}")
    print(benchmark_df.head(5))

    print("\n--- 3. Feedback Balance & Semantics ---")
    pos_count = (benchmark_df["is_recommended"] == True).sum()
    neg_count = (benchmark_df["is_recommended"] == False).sum()
    print(f"Positive interactions: {pos_count:,} ({pos_count/len(benchmark_df)*100:.2f}%)")
    print(f"Negative interactions: {neg_count:,} ({neg_count/len(benchmark_df)*100:.2f}%)")

if __name__ == "__main__":
    main()
