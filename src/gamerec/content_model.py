"""
GameRec Content-Based Filtering Recommender.
Extracts multi-modal TF-IDF features from community tags, game descriptions, and titles,
combining them via documented modality weights into a memory-efficient sparse CSR matrix.
Supports single-item similarity and multi-item user preference profile recommendations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Tuple
import joblib
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger("gamerec.content")


class ContentBasedRecommender:
    """
    Memory-efficient Sparse Content-Based Recommender.
    
    Feature Sources & Default Modality Weights:
    - w_tags (0.60): TF-IDF on Steam community tags (multi-word tags preserved via token standardization).
    - w_description (0.25): Sublinear TF-IDF on store synopsis (unigrams & bigrams, max 25,000 features).
    - w_title (0.15): Character/word n-gram TF-IDF on game titles (captures franchise and sequel linkages).
    
    Similarity Formulation:
    CosineSim(i, j) = X_combined[i] @ X_combined[j].T
    Because each component submatrix is L2-normalized and sum(weights) == 1,
    the inner product strictly equals:
    w_tags * cos(tags) + w_desc * cos(desc) + w_title * cos(title)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        max_desc_features: int = 25_000,
    ):
        if weights is None:
            self.weights = {"tags": 0.60, "description": 0.25, "title": 0.15}
        else:
            total_w = sum(weights.values())
            self.weights = {k: v / total_w for k, v in weights.items()}

        self.max_desc_features = max_desc_features
        self.is_fitted = False

        # Vectorizers
        self.tfidf_tags: Optional[TfidfVectorizer] = None
        self.tfidf_desc: Optional[TfidfVectorizer] = None
        self.tfidf_title: Optional[TfidfVectorizer] = None

        # Feature representations
        self.feature_matrix: Optional[csr_matrix] = None
        self.app_ids: np.ndarray = np.array([], dtype=np.uint32)
        self.app_id_to_idx: Dict[int, int] = {}
        self.catalog_data: Dict[str, Any] = {}

    def fit(self, catalog_df: pl.DataFrame) -> "ContentBasedRecommender":
        """
        Fits TF-IDF vectorizers and constructs the sparse L2-normalized feature matrix.
        """
        logger.info(f"Fitting ContentBasedRecommender on {catalog_df.shape[0]:,} catalog games...")

        self.app_ids = catalog_df["app_id"].to_numpy().astype(np.uint32)
        self.app_id_to_idx = {int(aid): idx for idx, aid in enumerate(self.app_ids)}

        # Cache key metadata in memory for fast explanation generation
        self.catalog_data = {
            "title": catalog_df["title"].to_list(),
            "tags": catalog_df["tags"].to_list(),
            "positive_ratio": catalog_df["positive_ratio"].to_list(),
            "user_reviews": catalog_df["user_reviews"].to_list(),
            "price_final": catalog_df["price_final"].to_list(),
            "bayesian_rating": catalog_df["bayesian_rating"].to_list(),
        }

        # 1. Tags Feature Matrix
        logger.info("Extracting tags TF-IDF representations...")
        tags_raw = catalog_df["tags"].to_list()
        # Join tags with underscore so multi-word tags (e.g. 'Open World') remain atomic
        tags_text = [
            " ".join([t.strip().replace(" ", "_") for t in tags]) if tags else ""
            for tags in tags_raw
        ]
        self.tfidf_tags = TfidfVectorizer(token_pattern=r"(?u)\b\S+\b", min_df=2)
        X_tags = self.tfidf_tags.fit_transform(tags_text)
        X_tags = normalize(X_tags, norm="l2")

        # 2. Description Feature Matrix
        logger.info("Extracting descriptions TF-IDF representations...")
        desc_text = catalog_df["description"].to_list()
        self.tfidf_desc = TfidfVectorizer(
            max_features=self.max_desc_features,
            stop_words="english",
            min_df=3,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X_desc = self.tfidf_desc.fit_transform(desc_text)
        X_desc = normalize(X_desc, norm="l2")

        # 3. Title Feature Matrix
        logger.info("Extracting title representations for franchise/sequel linking...")
        title_text = catalog_df["title"].str.replace_all(r"[^\w\s]", " ").to_list()
        self.tfidf_title = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2),
        )
        X_title = self.tfidf_title.fit_transform(title_text)
        X_title = normalize(X_title, norm="l2")

        # 4. Weighted Combination
        w_t = np.sqrt(self.weights["tags"])
        w_d = np.sqrt(self.weights["description"])
        w_m = np.sqrt(self.weights["title"])

        logger.info(
            f"Stacking sparse features with weights: tags={self.weights['tags']:.2f}, "
            f"desc={self.weights['description']:.2f}, title={self.weights['title']:.2f}..."
        )
        self.feature_matrix = hstack([w_t * X_tags, w_d * X_desc, w_m * X_title]).tocsr()

        mem_mb = self.feature_matrix.data.nbytes / (1024 * 1024)
        logger.info(
            f"Content feature matrix created: {self.feature_matrix.shape[0]:,} games x "
            f"{self.feature_matrix.shape[1]:,} features ({mem_mb:.2f} MB in RAM)."
        )

        self.is_fitted = True
        return self

    def recommend_similar(
        self,
        app_id: int,
        k: int = 10,
        exclude_seen: Optional[Union[Set[int], List[int]]] = None,
        min_reviews: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-K games most similar to a single seed game.
        """
        if not self.is_fitted or self.feature_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if app_id not in self.app_id_to_idx:
            raise ValueError(f"Game ID {app_id} not found in catalog.")

        if k <= 0:
            return []

        query_idx = self.app_id_to_idx[app_id]
        query_vec = self.feature_matrix[query_idx]

        # Fast sparse matrix-vector dot product across all 50,872 items
        sim_scores = (self.feature_matrix * query_vec.T).toarray().flatten()

        # Mask query game itself
        sim_scores[query_idx] = -1.0

        # Mask excluded games
        if exclude_seen:
            for ex_id in exclude_seen:
                if ex_id in self.app_id_to_idx:
                    sim_scores[self.app_id_to_idx[ex_id]] = -1.0

        # Optional review volume filter
        if min_reviews > 0:
            user_reviews = np.array(self.catalog_data["user_reviews"])
            sim_scores[user_reviews < min_reviews] = -1.0

        # Retrieve top K
        top_k_indices = np.argsort(sim_scores)[::-1][:k]

        query_title = self.catalog_data["title"][query_idx]
        query_tags = set(self.catalog_data["tags"][query_idx])

        recommendations = []
        for rank, idx in enumerate(top_k_indices, start=1):
            cand_score = float(sim_scores[idx])
            if cand_score <= 0.0:
                break

            cand_aid = int(self.app_ids[idx])
            cand_title = self.catalog_data["title"][idx]
            cand_tags = self.catalog_data["tags"][idx]
            cand_tags_set = set(cand_tags)
            shared_tags = sorted(list(query_tags.intersection(cand_tags_set)))

            explanation = self._build_item_explanation(
                query_title=query_title,
                candidate_title=cand_title,
                shared_tags=shared_tags,
                score=cand_score,
                rank=rank,
            )

            recommendations.append({
                "app_id": cand_aid,
                "title": cand_title,
                "rank": rank,
                "similarity_score": round(cand_score, 4),
                "shared_tags": shared_tags,
                "all_tags": cand_tags[:8],
                "positive_ratio": self.catalog_data["positive_ratio"][idx],
                "user_reviews": self.catalog_data["user_reviews"][idx],
                "price": self.catalog_data["price_final"][idx],
                "explanation": explanation,
            })

        return recommendations

    def recommend_user_profile(
        self,
        liked_games: List[Union[int, Dict[str, Any]]],
        k: int = 10,
        exclude_seen: Optional[Union[Set[int], List[int]]] = None,
        min_reviews: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-K games based on a composite multi-item user preference profile.
        
        Args:
            liked_games: List of app_ids (int) or dicts with {'app_id': int, 'weight': float}.
            k: Number of recommendations.
            exclude_seen: Set of app_ids to exclude.
            min_reviews: Minimum user review count.
        """
        if not self.is_fitted or self.feature_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if not liked_games:
            return []

        # Parse profile items and weights
        profile_indices = []
        profile_weights = []
        seen_ids = set()

        for item in liked_games:
            if isinstance(item, dict):
                aid = item.get("app_id")
                w = float(item.get("weight", 1.0))
            else:
                aid = int(item)
                w = 1.0

            if aid in self.app_id_to_idx:
                idx = self.app_id_to_idx[aid]
                profile_indices.append(idx)
                profile_weights.append(max(w, 0.01))
                seen_ids.add(aid)
            else:
                logger.warning(f"Liked game ID {aid} not in catalog; skipping.")

        if not profile_indices:
            return []

        # Compute weighted composite preference vector: sum(w_i * X[i])
        user_vec = csr_matrix((1, self.feature_matrix.shape[1]), dtype=np.float32)
        for idx, w in zip(profile_indices, profile_weights):
            user_vec = user_vec + (w * self.feature_matrix[idx])

        # Normalize preference vector to unit sphere
        user_vec = normalize(user_vec, norm="l2")

        # Compute cosine similarity across catalog
        sim_scores = (self.feature_matrix * user_vec.T).toarray().flatten()

        # Mask already liked games
        for aid in seen_ids:
            sim_scores[self.app_id_to_idx[aid]] = -1.0

        # Mask external excluded games
        if exclude_seen:
            for ex_id in exclude_seen:
                if ex_id in self.app_id_to_idx:
                    sim_scores[self.app_id_to_idx[ex_id]] = -1.0

        if min_reviews > 0:
            user_reviews = np.array(self.catalog_data["user_reviews"])
            sim_scores[user_reviews < min_reviews] = -1.0

        top_k_indices = np.argsort(sim_scores)[::-1][:k]

        # Gather user profile tags
        user_profile_tags = set()
        for idx in profile_indices:
            user_profile_tags.update(self.catalog_data["tags"][idx])

        recommendations = []
        for rank, idx in enumerate(top_k_indices, start=1):
            cand_score = float(sim_scores[idx])
            if cand_score <= 0.0:
                break

            cand_aid = int(self.app_ids[idx])
            cand_title = self.catalog_data["title"][idx]
            cand_tags = self.catalog_data["tags"][idx]
            shared_tags = sorted(list(user_profile_tags.intersection(set(cand_tags))))

            liked_titles = [self.catalog_data["title"][p_idx] for p_idx in profile_indices[:3]]
            explanation = (
                f"Matches your taste profile based on liked games ({', '.join(liked_titles)}) "
                f"with {len(shared_tags)} overlapping community tags "
                f"({', '.join(shared_tags[:5])}{'...' if len(shared_tags) > 5 else ''}) "
                f"and matching thematic descriptions (Score: {cand_score:.3f})."
            )

            recommendations.append({
                "app_id": cand_aid,
                "title": cand_title,
                "rank": rank,
                "similarity_score": round(cand_score, 4),
                "shared_tags": shared_tags,
                "all_tags": cand_tags[:8],
                "positive_ratio": self.catalog_data["positive_ratio"][idx],
                "user_reviews": self.catalog_data["user_reviews"][idx],
                "price": self.catalog_data["price_final"][idx],
                "explanation": explanation,
            })

        return recommendations

    def _build_item_explanation(
        self,
        query_title: str,
        candidate_title: str,
        shared_tags: List[str],
        score: float,
        rank: int,
    ) -> str:
        """Constructs a factual, metadata-backed explanation string."""
        if shared_tags:
            tag_str = ", ".join(shared_tags[:5])
            more = f" and {len(shared_tags) - 5} more" if len(shared_tags) > 5 else ""
            return (
                f"Recommended for players of '{query_title}'. Shares key themes & tags: "
                f"[{tag_str}{more}] with high semantic description similarity (Similarity: {score:.3f})."
            )
        else:
            return (
                f"Recommended for players of '{query_title}' based on high title/franchise "
                f"and genre-level description similarity (Similarity: {score:.3f})."
            )

    def save(self, file_path: Union[str, Path]) -> None:
        """Persists fitted model artifacts to disk."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Saved ContentBasedRecommender to {file_path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ContentBasedRecommender":
        """Loads a persisted model from disk."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"ContentBasedRecommender model file not found at {file_path}")
        model = joblib.load(file_path)
        logger.info(f"Loaded ContentBasedRecommender from {file_path}")
        return model
