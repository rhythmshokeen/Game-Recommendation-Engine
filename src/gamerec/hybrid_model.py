"""
GameRec Hybrid Recommendation Engine.
Combines Content-Based Filtering, Collaborative Filtering (Item-Item / SVD),
and Popularity Baselines with score calibration, configurable alpha parameter,
cold-start routing, seen-item exclusion, and diversity reranking.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import joblib
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from gamerec.content_model import ContentBasedRecommender
from gamerec.collaborative_model import (
    ItemItemCollaborativeRecommender,
    MatrixFactorizationRecommender,
)
from gamerec.popularity_model import PopularityRecommender
from gamerec.ranking import (
    min_max_scale,
    combine_hybrid_scores,
    maximal_marginal_relevance,
)

logger = logging.getLogger("gamerec.hybrid")


class HybridRecommender:
    """
    Configurable Hybrid Recommendation Engine.
    
    Architecture:
    Score(i) = alpha * Score_Content(i) + (1 - alpha) * Score_Collab(i)
    
    Parameters:
      - alpha = 1.0: Pure Content-Based Filtering
      - alpha = 0.0: Pure Collaborative Filtering
      - 0.0 < alpha < 1.0: Weighted Hybrid Blend (Default: 0.50)
      
    Cold-Start Routing Rules:
      - New User (0 interactions): Returns Popularity Baseline with Bayesian shrinkage.
      - Limited History User (< min_history_for_hybrid): Content-Based Profile with Popularity fallback.
      - Warm User (>= min_history_for_hybrid): Calibrated Hybrid Ranking.
      - New / Unreviewed Item: Content-Based similarity (collaborative score defaults to 0).
    """

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        collaborative_model: Union[ItemItemCollaborativeRecommender, MatrixFactorizationRecommender],
        popularity_model: PopularityRecommender,
        default_alpha: float = 0.5,
        min_history_for_hybrid: int = 3,
        diversity_lambda: float = 0.7,
        max_per_franchise: int = 2,
    ):
        if not (0.0 <= default_alpha <= 1.0):
            raise ValueError(f"default_alpha must be in [0.0, 1.0]; got {default_alpha}")

        self.content_model = content_model
        self.collab_model = collaborative_model
        self.pop_model = popularity_model
        self.default_alpha = default_alpha
        self.min_history_for_hybrid = min_history_for_hybrid
        self.diversity_lambda = diversity_lambda
        self.max_per_franchise = max_per_franchise

        # Cache catalog lookups
        self.catalog_data = content_model.catalog_data
        self.app_ids = content_model.app_ids
        self.app_id_to_idx = content_model.app_id_to_idx
        self.n_catalog = len(self.app_ids)

        # Precompute mapping from collaborative item index to catalog item index
        if hasattr(self.collab_model, "idx_to_app") and self.app_id_to_idx:
            self.collab_to_cat = np.array([
                self.app_id_to_idx.get(self.collab_model.idx_to_app[idx], -1)
                for idx in range(len(self.collab_model.idx_to_app))
            ], dtype=np.int32)
            self.collab_valid_mask = self.collab_to_cat >= 0
        else:
            self.collab_to_cat = np.array([], dtype=np.int32)
            self.collab_valid_mask = np.array([], dtype=bool)

    def recommend_user(
        self,
        user_id: int,
        k: int = 10,
        alpha: Optional[float] = None,
        exclude_seen: Optional[Union[Set[int], List[int]]] = None,
        use_diversity: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-K games for a registered user with cold-start routing and hybrid scoring.
        """
        if k <= 0:
            return []

        if alpha is None:
            alpha = self.default_alpha
        elif not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0; got {alpha}")

        exclude_set = set(exclude_seen) if exclude_seen else set()

        # Step 1: Inspect User Interaction History
        user_history = self.collab_model.user_history.get(user_id, set())
        exclude_set.update(user_history)

        # Cold-Start Rule A: Completely new user (0 interactions)
        if len(user_history) == 0:
            logger.info(f"User {user_id} has 0 history -> Routing to Popularity Baseline.")
            return self._handle_cold_start_user(k=k, exclude_seen=exclude_set)

        # Cold-Start Rule B: Limited history user (< min_history_for_hybrid)
        if len(user_history) < self.min_history_for_hybrid:
            logger.info(
                f"User {user_id} has {len(user_history)} interactions (< {self.min_history_for_hybrid}) -> "
                f"Routing to Content-Based taste profile."
            )
            return self._handle_limited_history_user(
                history_ids=list(user_history),
                k=k,
                exclude_seen=exclude_set,
                use_diversity=use_diversity,
            )

        # Step 2: Vectorized Content Scoring
        p_indices = [self.app_id_to_idx[a] for a in user_history if a in self.app_id_to_idx]
        if p_indices and self.content_model.feature_matrix is not None:
            u_vec = np.asarray(self.content_model.feature_matrix[p_indices].sum(axis=0)).flatten()
            norm = np.linalg.norm(u_vec)
            if norm > 0:
                u_vec = u_vec / norm
            c_raw = self.content_model.feature_matrix.dot(u_vec)
            c_scaled = min_max_scale(c_raw)
        else:
            c_scaled = np.zeros(self.n_catalog, dtype=np.float32)

        # Step 3: Vectorized Collaborative Scoring
        u_idx = self.collab_model.user_to_idx.get(user_id)
        cf_cat = np.zeros(self.n_catalog, dtype=np.float32)
        has_cf = np.zeros(self.n_catalog, dtype=bool)

        if u_idx is not None and self.collab_model.interaction_matrix is not None:
            user_vec = self.collab_model.interaction_matrix[u_idx]
            if isinstance(self.collab_model, ItemItemCollaborativeRecommender):
                raw_scores = (user_vec @ self.collab_model.similarity_matrix).toarray().flatten()
            elif isinstance(self.collab_model, MatrixFactorizationRecommender):
                user_vec_norm = normalize(user_vec, norm="l2", axis=1)
                user_latent = (user_vec_norm @ self.collab_model.item_factors).flatten()
                user_latent_norm = user_latent / (np.linalg.norm(user_latent) + 1e-8)
                raw_scores = self.collab_model.item_factors @ user_latent_norm
            else:
                raw_scores = np.zeros(len(self.collab_to_cat), dtype=np.float32)

            cf_scaled = min_max_scale(raw_scores)
            if len(self.collab_to_cat) > 0:
                cf_cat[self.collab_to_cat[self.collab_valid_mask]] = cf_scaled[self.collab_valid_mask]
                has_cf[self.collab_to_cat[self.collab_valid_mask]] = True

        # Step 4: Combine with configurable alpha
        if alpha == 1.0:
            h_scores = c_scaled.copy()
        elif alpha == 0.0:
            h_scores = cf_cat.copy()
        else:
            h_scores = np.where(has_cf, (alpha * c_scaled) + ((1.0 - alpha) * cf_cat), alpha * c_scaled)

        # Step 5: Mask excluded/seen games
        for aid in exclude_set:
            if aid in self.app_id_to_idx:
                h_scores[self.app_id_to_idx[aid]] = -1.0

        # Step 6: Select Top Candidates
        pool_size = min(k * 4, self.n_catalog)
        valid_pos = np.where(h_scores > 0.0)[0]
        if len(valid_pos) == 0:
            return self._handle_cold_start_user(k=k, exclude_seen=exclude_set)

        if len(valid_pos) > pool_size:
            top_part = np.argpartition(-h_scores[valid_pos], pool_size)[:pool_size]
            candidate_indices = valid_pos[top_part]
            candidate_indices = candidate_indices[np.argsort(-h_scores[candidate_indices])]
        else:
            candidate_indices = valid_pos[np.argsort(-h_scores[valid_pos])]

        candidate_dicts = []
        for idx in candidate_indices:
            aid = int(self.app_ids[idx])
            h_score = float(h_scores[idx])
            c_score = float(c_scaled[idx])
            cf_score = float(cf_cat[idx])
            candidate_dicts.append({
                "app_id": aid,
                "title": self.catalog_data["title"][idx],
                "score": h_score,
                "hybrid_score": h_score,
                "content_score": c_score,
                "collab_score": cf_score,
                "alpha": alpha,
                "all_tags": self.catalog_data["tags"][idx][:8],
                "positive_ratio": self.catalog_data["positive_ratio"][idx],
                "user_reviews": self.catalog_data["user_reviews"][idx],
                "price": self.catalog_data["price_final"][idx],
                "explanation": self._build_hybrid_explanation(
                    aid=aid,
                    title=self.catalog_data["title"][idx],
                    h_score=h_score,
                    c_score=c_score,
                    cf_score=cf_score,
                    alpha=alpha,
                    user_history=user_history,
                ),
            })

        # Step 6: Optional Diversity Reranking
        if use_diversity and self.diversity_lambda < 1.0:
            final_recs = maximal_marginal_relevance(
                candidates=candidate_dicts,
                feature_matrix=self.content_model.feature_matrix,
                app_to_idx=self.app_id_to_idx,
                diversity_lambda=self.diversity_lambda,
                top_k=k,
                max_per_franchise=self.max_per_franchise,
            )
        else:
            final_recs = candidate_dicts[:k]
            for r_idx, item in enumerate(final_recs, start=1):
                item["rank"] = r_idx

        return final_recs

    def recommend_interactive(
        self,
        seed_games: List[Union[int, Dict[str, Any]]],
        k: int = 10,
        alpha: Optional[float] = None,
        exclude_seen: Optional[Union[Set[int], List[int]]] = None,
        use_diversity: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Interactive multi-modal recommendation for guest sessions or users with selected preferences.
        """
        if k <= 0:
            return []

        if not seed_games:
            return self._handle_cold_start_user(k=k, exclude_seen=exclude_seen)

        if alpha is None:
            alpha = self.default_alpha
        elif not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0; got {alpha}")

        exclude_set = set(exclude_seen) if exclude_seen else set()

        # Parse seed games
        seed_ids = []
        for g in seed_games:
            aid = g["app_id"] if isinstance(g, dict) else int(g)
            seed_ids.append(aid)
            exclude_set.add(aid)

        # 1. Content scores
        content_scores_dict = self._compute_content_profile_scores(seed_games)

        # 2. Collaborative scores via on-the-fly projection
        collab_scores_dict = self._compute_collaborative_items_scores(seed_ids)

        # 3. Combine scores
        candidate_app_ids = list(self.app_ids)
        combined = combine_hybrid_scores(
            candidate_app_ids=candidate_app_ids,
            content_scores_dict=content_scores_dict,
            collab_scores_dict=collab_scores_dict,
            alpha=alpha,
        )

        for aid in exclude_set:
            if aid in combined:
                combined[aid]["hybrid_score"] = -1.0

        sorted_candidates = sorted(
            [
                (aid, scores["hybrid_score"], scores["content_score"], scores["collab_score"])
                for aid, scores in combined.items()
                if scores["hybrid_score"] > 0.0
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        if not sorted_candidates:
            return self._handle_cold_start_user(k=k, exclude_seen=exclude_set)

        pool_size = min(k * 4, len(sorted_candidates))
        candidate_dicts = []
        for aid, h_score, c_score, cf_score in sorted_candidates[:pool_size]:
            idx = self.app_id_to_idx[aid]
            candidate_dicts.append({
                "app_id": aid,
                "title": self.catalog_data["title"][idx],
                "score": h_score,
                "hybrid_score": h_score,
                "content_score": c_score,
                "collab_score": cf_score,
                "alpha": alpha,
                "all_tags": self.catalog_data["tags"][idx][:8],
                "positive_ratio": self.catalog_data["positive_ratio"][idx],
                "user_reviews": self.catalog_data["user_reviews"][idx],
                "price": self.catalog_data["price_final"][idx],
                "explanation": self._build_hybrid_explanation(
                    aid=aid,
                    title=self.catalog_data["title"][idx],
                    h_score=h_score,
                    c_score=c_score,
                    cf_score=cf_score,
                    alpha=alpha,
                    user_history=set(seed_ids),
                ),
            })

        if use_diversity and self.diversity_lambda < 1.0:
            final_recs = maximal_marginal_relevance(
                candidates=candidate_dicts,
                feature_matrix=self.content_model.feature_matrix,
                app_to_idx=self.app_id_to_idx,
                diversity_lambda=self.diversity_lambda,
                top_k=k,
                max_per_franchise=self.max_per_franchise,
            )
        else:
            final_recs = candidate_dicts[:k]
            for r_idx, item in enumerate(final_recs, start=1):
                item["rank"] = r_idx

        return final_recs

    def _compute_content_profile_scores(
        self, liked_items: List[Union[int, Dict[str, Any]]]
    ) -> Dict[int, float]:
        """Computes min-max scaled content scores across all catalog items."""
        profile_indices = []
        profile_weights = []

        for item in liked_items:
            aid = item["app_id"] if isinstance(item, dict) else int(item)
            w = float(item.get("weight", 1.0)) if isinstance(item, dict) else 1.0
            if aid in self.app_id_to_idx:
                profile_indices.append(self.app_id_to_idx[aid])
                profile_weights.append(max(w, 0.01))

        if not profile_indices:
            return {}

        # Composite user taste vector: sum(w_i * X_i)
        user_vec = csr_matrix((1, self.content_model.feature_matrix.shape[1]), dtype=np.float32)
        for idx, w in zip(profile_indices, profile_weights):
            user_vec = user_vec + (w * self.content_model.feature_matrix[idx])

        user_vec = normalize(user_vec, norm="l2")
        raw_sims = (self.content_model.feature_matrix * user_vec.T).toarray().flatten()

        # Scale non-negative similarities
        scaled_sims = min_max_scale(raw_sims)
        return {int(aid): float(scaled_sims[i]) for i, aid in enumerate(self.app_ids)}

    def _compute_collaborative_user_scores(self, user_id: int) -> Dict[int, float]:
        """Computes min-max scaled collaborative filtering scores across active items."""
        if user_id not in self.collab_model.user_to_idx:
            return {}

        u_idx = self.collab_model.user_to_idx[user_id]
        user_vec = self.collab_model.interaction_matrix[u_idx]

        if isinstance(self.collab_model, ItemItemCollaborativeRecommender):
            raw_scores = (user_vec @ self.collab_model.similarity_matrix).toarray().flatten()
        elif isinstance(self.collab_model, MatrixFactorizationRecommender):
            user_vec_norm = normalize(user_vec, norm="l2", axis=1)
            user_latent = (user_vec_norm @ self.collab_model.item_factors).flatten()
            user_latent_norm = user_latent / (np.linalg.norm(user_latent) + 1e-8)
            raw_scores = self.collab_model.item_factors @ user_latent_norm
        else:
            return {}

        scaled_scores = min_max_scale(raw_scores)
        return {
            int(self.collab_model.idx_to_app[idx]): float(scaled_scores[idx])
            for idx in range(len(scaled_scores))
        }

    def _compute_collaborative_items_scores(self, history_ids: List[int]) -> Dict[int, float]:
        """Computes on-the-fly collaborative scores for an arbitrary item list."""
        valid_indices = [
            self.collab_model.app_to_idx[a]
            for a in history_ids
            if a in self.collab_model.app_to_idx
        ]
        if not valid_indices:
            return {}

        n_collab_items = len(self.collab_model.app_to_idx)

        if isinstance(self.collab_model, ItemItemCollaborativeRecommender):
            user_vec = csr_matrix(
                ([1.0] * len(valid_indices), ([0] * len(valid_indices), valid_indices)),
                shape=(1, n_collab_items),
                dtype=np.float32,
            )
            raw_scores = (user_vec @ self.collab_model.similarity_matrix).toarray().flatten()
        elif isinstance(self.collab_model, MatrixFactorizationRecommender):
            user_latent = self.collab_model.item_factors[valid_indices].mean(axis=0)
            user_latent_norm = user_latent / (np.linalg.norm(user_latent) + 1e-8)
            raw_scores = self.collab_model.item_factors @ user_latent_norm
        else:
            return {}

        scaled_scores = min_max_scale(raw_scores)
        return {
            int(self.collab_model.idx_to_app[idx]): float(scaled_scores[idx])
            for idx in range(len(scaled_scores))
        }

    def _handle_cold_start_user(
        self,
        k: int = 10,
        exclude_seen: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Gracefully routes cold-start users to Bayesian Popularity baseline."""
        pop_recs = self.pop_model.recommend(
            k=k,
            exclude_app_ids=exclude_seen,
            ranking_metric="weighted_popularity",
        )
        for rank, r in enumerate(pop_recs, start=1):
            r["rank"] = rank
            r["score"] = r["score"]
            r["hybrid_score"] = r["score"]
            r["content_score"] = 0.0
            r["collab_score"] = 0.0
            r["alpha"] = self.default_alpha
            r["explanation"] = (
                f"Community baseline recommendation for new users (Bayesian quality: {r['bayesian_rating']:.3f}, "
                f"{r['positive_ratio']}% positive on {r['user_reviews']:,} reviews)."
            )
        return pop_recs

    def _handle_limited_history_user(
        self,
        history_ids: List[int],
        k: int = 10,
        exclude_seen: Optional[Set[int]] = None,
        use_diversity: bool = True,
    ) -> List[Dict[str, Any]]:
        """Routes users with < min_history_for_hybrid to Content-Based taste matching."""
        content_recs = self.content_model.recommend_user_profile(
            liked_games=history_ids,
            k=k,
            exclude_seen=exclude_seen,
        )
        if not content_recs:
            return self._handle_cold_start_user(k=k, exclude_seen=exclude_seen)

        for rank, r in enumerate(content_recs, start=1):
            r["rank"] = rank
            r["hybrid_score"] = r["similarity_score"]
            r["content_score"] = r["similarity_score"]
            r["collab_score"] = 0.0
            r["alpha"] = 1.0
            r["explanation"] = (
                f"Content match for emerging user history: shares key tags "
                f"[{', '.join(r['shared_tags'][:4])}] with played games (Score: {r['similarity_score']:.3f})."
            )
        return content_recs

    def _build_hybrid_explanation(
        self,
        aid: int,
        title: str,
        h_score: float,
        c_score: float,
        cf_score: float,
        alpha: float,
        user_history: Set[int],
    ) -> str:
        """Constructs a factual, transparent, and calibrated explanation."""
        if alpha == 1.0:
            return (
                f"Content-only recommendation: High thematic similarity ({c_score:.3f}) "
                f"with your played titles."
            )
        elif alpha == 0.0:
            return (
                f"Collaborative-only recommendation: High community co-play affinity "
                f"({cf_score:.3f}) from players with similar libraries."
            )
        else:
            # Hybrid
            return (
                f"Hybrid match (Score: {h_score:.3f}, α={alpha:.2f}): Blends thematic tag similarity "
                f"({c_score:.3f}) with collaborative co-play patterns ({cf_score:.3f}) from your library."
            )

    def save_config(self, file_path: Union[str, Path]) -> None:
        """Saves engine hyperparameters for reproducible inference."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "default_alpha": self.default_alpha,
            "min_history_for_hybrid": self.min_history_for_hybrid,
            "diversity_lambda": self.diversity_lambda,
            "max_per_franchise": self.max_per_franchise,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved hybrid configuration to {file_path}")
