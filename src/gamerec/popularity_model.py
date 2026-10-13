"""
GameRec Popularity Baseline Recommender.
Provides confidence-weighted and Bayesian-adjusted popularity rankings
for overall catalog and genre/tag-specific slices with explainability.
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any
import joblib
import numpy as np
import polars as pl

logger = logging.getLogger("gamerec.popularity")


class PopularityRecommender:
    """
    Confidence-aware Bayesian Popularity Baseline Model.
    
    Mathematical Formulation:
    1. Bayesian Shrinkage Rating (W):
       W = (v / (v + m)) * R + (m / (v + m)) * C
       where:
         - R = observed positive review ratio (positive_ratio / 100)
         - v = user review volume (user_reviews)
         - C = prior mean positive review ratio across all training games
         - m = minimum-support regularization threshold (computed from training data distribution)

    2. Weighted Popularity Score (PopScore):
       PopScore = W * log10(v + 1)
       Balances qualitative satisfaction (W) with social engagement volume (log10(v + 1)).
    """

    def __init__(self, support_quantile: float = 0.75, default_min_reviews: int = 50):
        self.support_quantile = support_quantile
        self.default_min_reviews = default_min_reviews
        self.is_fitted = False
        
        # Fitted parameters
        self.prior_mean_positive_ratio: float = 0.77
        self.regularization_m: float = 200.0
        self.min_support_threshold: int = 50
        
        # Stored data structures
        self.catalog_df: Optional[pl.DataFrame] = None
        self.app_id_to_row: Dict[int, int] = {}
        self.tag_index: Dict[str, List[int]] = {}

    def fit(
        self,
        catalog_df: pl.DataFrame,
        min_support_threshold: Optional[int] = None,
    ) -> "PopularityRecommender":
        """
        Fits popularity prior parameters strictly from the training catalog data.
        """
        logger.info("Fitting PopularityRecommender on catalog data...")
        self.catalog_df = catalog_df.clone()

        # Compute prior mean C from training games with reviews
        reviewed_games = self.catalog_df.filter(pl.col("user_reviews") > 0)
        if reviewed_games.shape[0] == 0:
            raise ValueError("No games with reviews found in catalog!")

        self.prior_mean_positive_ratio = float(reviewed_games["positive_ratio"].mean() / 100.0)

        # Compute regularization threshold m from training distribution
        review_volumes = reviewed_games["user_reviews"].to_numpy().astype(float)
        self.regularization_m = float(np.quantile(review_volumes, self.support_quantile))

        if min_support_threshold is not None:
            self.min_support_threshold = min_support_threshold
        else:
            self.min_support_threshold = self.default_min_reviews

        # Calculate scores
        v = self.catalog_df["user_reviews"].to_numpy().astype(float)
        r = self.catalog_df["positive_ratio"].to_numpy().astype(float) / 100.0
        m = self.regularization_m
        c = self.prior_mean_positive_ratio

        # Bayesian rating: shrinkage toward catalog prior C
        bayesian_w = (v / (v + m)) * r + (m / (v + m)) * c
        # Weighted popularity: Bayesian quality * log10(volume + 1)
        weighted_pop = bayesian_w * np.log10(v + 1.0)

        self.catalog_df = self.catalog_df.with_columns([
            pl.Series("computed_bayesian_rating", bayesian_w.astype(np.float32)),
            pl.Series("weighted_pop_score", weighted_pop.astype(np.float32)),
        ])

        # Build app_id lookup and tag index
        self.app_id_to_row = {
            int(aid): idx for idx, aid in enumerate(self.catalog_df["app_id"].to_list())
        }

        # Index tags for fast O(1) genre filtering
        self.tag_index = {}
        for idx, tags_list in enumerate(self.catalog_df["tags"].to_list()):
            if tags_list:
                for t in tags_list:
                    tag_clean = t.strip().lower()
                    if tag_clean not in self.tag_index:
                        self.tag_index[tag_clean] = []
                    self.tag_index[tag_clean].append(idx)

        self.is_fitted = True
        logger.info(
            f"PopularityRecommender fitted successfully. "
            f"Prior Mean C={self.prior_mean_positive_ratio:.4f}, Regularization m={self.regularization_m:.1f}, "
            f"Indexed tags={len(self.tag_index)}"
        )
        return self

    def recommend(
        self,
        k: int = 10,
        tag: Optional[str] = None,
        platform: Optional[str] = None,
        min_reviews: Optional[int] = None,
        exclude_app_ids: Optional[Union[Set[int], List[int]]] = None,
        ranking_metric: str = "weighted_popularity",
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-K popular games based on confidence-adjusted scoring.
        
        Args:
            k: Number of recommendations to return.
            tag: Optional community tag / genre filter (e.g. 'Roguelike', 'RPG', 'Action').
            platform: Optional platform filter ('win', 'mac', 'linux', 'steam_deck').
            min_reviews: Minimum review threshold (defaults to fitted threshold).
            exclude_app_ids: Set of game IDs to exclude (e.g. already seen games).
            ranking_metric: 'weighted_popularity' (default) or 'bayesian_rating' or 'user_reviews'.
        """
        if not self.is_fitted or self.catalog_df is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if k <= 0:
            return []

        if min_reviews is None:
            min_reviews = self.min_support_threshold

        exclude_set = set(exclude_app_ids) if exclude_app_ids else set()

        # Determine candidate rows
        if tag is not None:
            tag_clean = tag.strip().lower()
            if tag_clean not in self.tag_index:
                logger.warning(f"Tag '{tag}' not found in catalog tag index.")
                return []
            candidate_indices = self.tag_index[tag_clean]
            candidates = self.catalog_df[candidate_indices]
        else:
            candidates = self.catalog_df

        # Apply platform filters
        if platform is not None:
            plat_col = platform.strip().lower()
            if plat_col in ["win", "mac", "linux", "steam_deck"]:
                candidates = candidates.filter(pl.col(plat_col) == True)
            else:
                raise ValueError(f"Invalid platform: {platform}. Expected 'win', 'mac', 'linux', or 'steam_deck'.")

        # Apply minimum review filter
        candidates = candidates.filter(pl.col("user_reviews") >= min_reviews)

        # Exclude specified IDs
        if exclude_set:
            candidates = candidates.filter(~pl.col("app_id").is_in(list(exclude_set)))

        if candidates.shape[0] == 0:
            return []

        # Sort by selected metric
        sort_col = "weighted_pop_score"
        if ranking_metric == "bayesian_rating":
            sort_col = "computed_bayesian_rating"
        elif ranking_metric == "user_reviews":
            sort_col = "user_reviews"

        top_df = candidates.sort(sort_col, descending=True).head(k)

        results = []
        for rank, row in enumerate(top_df.to_dicts(), start=1):
            aid = row["app_id"]
            title = row["title"]
            v = row["user_reviews"]
            pos_ratio = row["positive_ratio"]
            bayes_score = row["computed_bayesian_rating"]
            pop_score = row["weighted_pop_score"]
            price = row["price_final"]
            tags = row["tags"]

            # Construct clear, metadata-backed explanation
            tag_context = f" in '{tag.title()}'" if tag else ""
            explanation = (
                f"Ranked #{rank}{tag_context} based on {pos_ratio}% positive reviews "
                f"across {v:,} total reviews (Bayesian quality score: {bayes_score:.3f}, "
                f"weighted popularity index: {pop_score:.2f})."
            )

            results.append({
                "app_id": aid,
                "title": title,
                "rank": rank,
                "score": float(pop_score if ranking_metric == "weighted_popularity" else bayes_score),
                "bayesian_rating": float(bayes_score),
                "positive_ratio": int(pos_ratio),
                "user_reviews": int(v),
                "price": float(price),
                "tags": tags,
                "explanation": explanation,
            })

        return results

    def get_available_tags(self, min_games: int = 10) -> List[str]:
        """Returns sorted list of available tags that meet minimum game count."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        valid_tags = [
            tag.title() for tag, indices in self.tag_index.items() if len(indices) >= min_games
        ]
        return sorted(valid_tags)

    def explain_model(self) -> Dict[str, Any]:
        """Provides mathematical details, parameters, strengths, and limitations."""
        return {
            "model_name": "Popularity Baseline (Bayesian Shrinkage)",
            "parameters": {
                "prior_mean_positive_ratio": self.prior_mean_positive_ratio,
                "regularization_threshold_m": self.regularization_m,
                "default_min_support_reviews": self.min_support_threshold,
            },
            "formula": "W = (v / (v + m)) * R + (m / (v + m)) * C; PopScore = W * log10(v + 1)",
            "strengths": [
                "Zero cold-start sensitivity for users (recommends globally recognized high-quality games).",
                "Regularizes niche games with small review samples toward the catalog mean.",
                "Computationally instantaneous (O(1) retrieval after sorting).",
            ],
            "limitations": [
                "Popularity Bias / Matthew Effect: Heavily favors dominant blockbusters (CS:GO, Terraria, Witcher 3).",
                "Zero Personalization: Provides identical recommendations to every user without considering individual taste.",
                "Item Cold-Start Failure: Completely overlooks new or unreviewed games (26% of catalog).",
            ],
        }

    def save(self, file_path: Union[str, Path]) -> None:
        """Persists fitted model state to disk using joblib."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Saved PopularityRecommender to {file_path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "PopularityRecommender":
        """Loads a persisted model from disk."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PopularityRecommender model file not found at {file_path}")
        model = joblib.load(file_path)
        logger.info(f"Loaded PopularityRecommender from {file_path}")
        return model
