"""
GameRec Collaborative Filtering Models.
Implements:
1. Item-Item Collaborative Filtering with Bayesian Similarity Shrinkage
2. Latent Factor Matrix Factorization (TruncatedSVD on Implicit Interaction Matrices)
3. Leak-free Temporal Train-Test Splitting
4. Seen-item exclusion and explanation generation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import joblib
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, csc_matrix, diags
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

logger = logging.getLogger("gamerec.collaborative")


def temporal_train_test_split(
    interactions_df: pl.DataFrame,
    min_user_interactions: int = 5,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Chronologically splits user interactions into training and evaluation holdout.
    For every eligible user (>= min_user_interactions), the latest interaction is placed
    into the test set, guaranteeing zero future-to-past data leakage.
    """
    logger.info("Executing temporal train-test split (leave-last-out holdout)...")
    
    # Filter users with sufficient interaction volume
    user_counts = (
        interactions_df
        .group_by("user_id")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_user_interactions)
    )
    eligible_users = set(user_counts["user_id"].to_list())

    filtered_df = interactions_df.filter(pl.col("user_id").is_in(list(eligible_users)))

    # Sort interactions strictly chronologically
    sorted_df = filtered_df.sort(["user_id", "date", "review_id"])

    # Rank interactions within user timeline
    with_rank = sorted_df.with_columns([
        pl.col("review_id").cum_count().over("user_id").alias("user_step"),
        pl.len().over("user_id").alias("user_total"),
    ])

    train_df = (
        with_rank
        .filter(pl.col("user_step") < pl.col("user_total"))
        .drop(["user_step", "user_total"])
    )
    test_df = (
        with_rank
        .filter(pl.col("user_step") == pl.col("user_total"))
        .drop(["user_step", "user_total"])
    )

    logger.info(
        f"Split complete: Train={train_df.shape[0]:,} interactions ({train_df['user_id'].n_unique():,} users), "
        f"Test={test_df.shape[0]:,} interactions ({test_df['user_id'].n_unique():,} users)."
    )
    return train_df, test_df


def build_interaction_matrix(
    interactions_df: pl.DataFrame,
    user_to_idx: Optional[Dict[int, int]] = None,
    app_to_idx: Optional[Dict[int, int]] = None,
    alpha_hours: float = 0.5,
    filter_positive_only: bool = True,
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Constructs a memory-efficient sparse CSR user-item interaction matrix.
    Incorporates implicit feedback confidence: c_ui = 1.0 + alpha * ln(1.0 + hours).
    """
    if filter_positive_only:
        data_df = interactions_df.filter(pl.col("is_recommended") == True)
    else:
        data_df = interactions_df

    # Build or reuse mappings
    if user_to_idx is None:
        unique_users = sorted(list(set(data_df["user_id"].to_list())))
        user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    if app_to_idx is None:
        unique_apps = sorted(list(set(data_df["app_id"].to_list())))
        app_to_idx = {aid: idx for idx, aid in enumerate(unique_apps)}

    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_app = {idx: aid for aid, idx in app_to_idx.items()}

    # Extract coordinates
    filtered_rows = data_df.filter(
        pl.col("user_id").is_in(list(user_to_idx.keys()))
        & pl.col("app_id").is_in(list(app_to_idx.keys()))
    )

    user_indices = [user_to_idx[u] for u in filtered_rows["user_id"].to_list()]
    app_indices = [app_to_idx[a] for a in filtered_rows["app_id"].to_list()]

    # Confidence calculation: 1.0 + alpha * ln(1 + hours)
    hours_arr = filtered_rows["hours"].to_numpy().astype(np.float32)
    conf_values = (1.0 + alpha_hours * np.log1p(hours_arr)).astype(np.float32)

    shape = (len(user_to_idx), len(app_to_idx))
    R = csr_matrix((conf_values, (user_indices, app_indices)), shape=shape, dtype=np.float32)

    return R, user_to_idx, app_to_idx, idx_to_user, idx_to_app


class ItemItemCollaborativeRecommender:
    """
    Item-Based Collaborative Filtering with Bayesian Similarity Shrinkage.
    
    Formulation:
    Given item interaction vectors r_{*i} and r_{*j}:
    S_{ij} = (r_{*i} @ r_{*j}) / ( ||r_{*i}||_2 * ||r_{*j}||_2 + lambda )
    where lambda >= 0 regularizes item pairs with sparse co-occurrence support.
    
    Prediction for user u:
    y_hat_u = r_u @ S
    """

    def __init__(
        self,
        top_k_neighbors: int = 100,
        shrinkage_lambda: float = 10.0,
        alpha_hours: float = 0.5,
    ):
        self.top_k_neighbors = top_k_neighbors
        self.shrinkage_lambda = shrinkage_lambda
        self.alpha_hours = alpha_hours
        self.is_fitted = False

        # Internal state
        self.similarity_matrix: Optional[csr_matrix] = None
        self.user_to_idx: Dict[int, int] = {}
        self.app_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.idx_to_app: Dict[int, int] = {}
        self.user_history: Dict[int, Set[int]] = {}
        self.catalog_titles: Dict[int, str] = {}
        self.interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        train_interactions: pl.DataFrame,
        catalog_df: Optional[pl.DataFrame] = None,
    ) -> "ItemItemCollaborativeRecommender":
        """
        Fits item-item similarity matrix strictly using training interactions.
        """
        logger.info("Fitting ItemItemCollaborativeRecommender on training interactions...")

        if catalog_df is not None:
            self.catalog_titles = dict(
                zip(catalog_df["app_id"].to_list(), catalog_df["title"].to_list())
            )

        # Build sparse interaction matrix
        R, u2i, a2i, i2u, i2a = build_interaction_matrix(
            train_interactions,
            alpha_hours=self.alpha_hours,
            filter_positive_only=True,
        )
        self.interaction_matrix = R
        self.user_to_idx = u2i
        self.app_to_idx = a2i
        self.idx_to_user = i2u
        self.idx_to_app = i2a

        # Store training history per user for fast exclusion
        pos_train = train_interactions.filter(pl.col("is_recommended") == True)
        self.user_history = {}
        for row in pos_train.select(["user_id", "app_id"]).iter_rows():
            uid, aid = int(row[0]), int(row[1])
            if uid not in self.user_history:
                self.user_history[uid] = set()
            self.user_history[uid].add(aid)

        # Compute Item-Item Cosine Similarity with Shrinkage
        n_items = len(self.app_to_idx)
        logger.info(f"Computing item-item similarity matrix for {n_items:,} items...")

        # Normalize column vectors (item vectors)
        item_matrix = R.tocsc()
        norms = np.sqrt(item_matrix.multiply(item_matrix).sum(axis=0)).A1.astype(np.float32)

        # Compute raw dot product: R.T @ R
        raw_sim = (item_matrix.T @ item_matrix).tocsr()

        # Apply shrinkage denominator: (norm_i * norm_j + lambda)
        if self.shrinkage_lambda > 0.0:
            # S_ij = dot_ij / (norm_i * norm_j + lambda)
            rows, cols = raw_sim.nonzero()
            data = raw_sim.data
            shrinkage_denoms = (norms[rows] * norms[cols]) + self.shrinkage_lambda
            shrinkage_denoms[shrinkage_denoms == 0] = 1.0
            shrunk_data = (data / shrinkage_denoms).astype(np.float32)
            sim_matrix = csr_matrix((shrunk_data, (rows, cols)), shape=(n_items, n_items))
        else:
            # Standard cosine similarity
            item_normed = normalize(item_matrix, norm="l2", axis=0)
            sim_matrix = (item_normed.T @ item_normed).tocsr()

        # Remove diagonal self-similarity
        sim_matrix.setdiag(0.0)
        sim_matrix.eliminate_zeros()

        # Prune to top-K neighbors per item to keep memory minimal
        if self.top_k_neighbors > 0:
            logger.info(f"Pruning similarity matrix to top-{self.top_k_neighbors} neighbors per item...")
            sim_matrix = self._prune_top_k(sim_matrix, self.top_k_neighbors)

        self.similarity_matrix = sim_matrix
        mem_mb = self.similarity_matrix.data.nbytes / (1024 * 1024)
        logger.info(f"Item-Item similarity matrix complete: {mem_mb:.2f} MB in RAM.")
        self.is_fitted = True
        return self

    def _prune_top_k(self, mat: csr_matrix, k: int) -> csr_matrix:
        """Retains only the top-K largest values per row in a sparse matrix."""
        mat_csr = mat.tocsr()
        new_data, new_indices, new_indptr = [], [], [0]

        for i in range(mat_csr.shape[0]):
            start, end = mat_csr.indptr[i], mat_csr.indptr[i + 1]
            row_data = mat_csr.data[start:end]
            row_indices = mat_csr.indices[start:end]

            if len(row_data) > k:
                top_idx = np.argpartition(row_data, -k)[-k:]
                # Sort the top K
                top_sorted = top_idx[np.argsort(-row_data[top_idx])]
                new_data.extend(row_data[top_sorted])
                new_indices.extend(row_indices[top_sorted])
                new_indptr.append(new_indptr[-1] + k)
            else:
                new_data.extend(row_data)
                new_indices.extend(row_indices)
                new_indptr.append(new_indptr[-1] + len(row_data))

        return csr_matrix(
            (new_data, new_indices, new_indptr), shape=mat.shape, dtype=np.float32
        )

    def recommend_user(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: Optional[Union[Set[int], List[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-K items for a known user using their training interaction vector.
        """
        if not self.is_fitted or self.similarity_matrix is None or self.interaction_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in training matrix. Insufficient history.")
            return []

        user_idx = self.user_to_idx[user_id]
        user_vec = self.interaction_matrix[user_idx]  # (1, n_items)

        # Score candidates: s = r_u @ S
        scores = (user_vec @ self.similarity_matrix).toarray().flatten()

        # Exclude seen items from training history
        seen_set = self.user_history.get(user_id, set()).copy()
        if exclude_seen:
            seen_set.update(exclude_seen)

        for aid in seen_set:
            if aid in self.app_to_idx:
                scores[self.app_to_idx[aid]] = -1.0

        top_indices = np.argsort(scores)[::-1][:k]

        recommendations = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                break
            aid = int(self.idx_to_app[idx])
            title = self.catalog_titles.get(aid, f"Game #{aid}")

            # Find top contributing history item for explanation
            cand_col = self.similarity_matrix[:, idx].toarray().flatten()
            user_inter = user_vec.toarray().flatten()
            contributions = user_inter * cand_col
            best_hist_idx = int(np.argmax(contributions))
            best_hist_aid = int(self.idx_to_app[best_hist_idx])
            best_hist_title = self.catalog_titles.get(best_hist_aid, f"Game #{best_hist_aid}")
            best_sim = float(cand_col[best_hist_idx])

            explanation = (
                f"Recommended because you played '{best_hist_title}' "
                f"(Item co-occurrence similarity: {best_sim:.3f})."
            )

            recommendations.append({
                "app_id": aid,
                "title": title,
                "rank": rank,
                "score": round(score, 4),
                "model": "Item-Item Collaborative",
                "explanation": explanation,
            })

        return recommendations

    def recommend_items(
        self,
        history_app_ids: List[int],
        k: int = 10,
        exclude_seen: Optional[Union[Set[int], List[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-K items on-the-fly for any arbitrary list of played games.
        Enables real-time collaborative recommendations for new users with known history.
        """
        if not self.is_fitted or self.similarity_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        valid_indices = [self.app_to_idx[a] for a in history_app_ids if a in self.app_to_idx]
        if not valid_indices:
            return []

        # Construct unit interaction vector efficiently from coordinates
        user_vec = csr_matrix(
            ([1.0] * len(valid_indices), ([0] * len(valid_indices), valid_indices)),
            shape=(1, len(self.app_to_idx)),
            dtype=np.float32,
        )

        scores = (user_vec @ self.similarity_matrix).toarray().flatten()

        seen_set = set(history_app_ids)
        if exclude_seen:
            seen_set.update(exclude_seen)

        for aid in seen_set:
            if aid in self.app_to_idx:
                scores[self.app_to_idx[aid]] = -1.0

        top_indices = np.argsort(scores)[::-1][:k]

        recommendations = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                break
            aid = int(self.idx_to_app[idx])
            title = self.catalog_titles.get(aid, f"Game #{aid}")

            # Top contributing item
            cand_col = self.similarity_matrix[:, idx].toarray().flatten()
            contribs = [cand_col[h_idx] for h_idx in valid_indices]
            best_pos = int(np.argmax(contribs))
            best_h_aid = history_app_ids[best_pos]
            best_title = self.catalog_titles.get(best_h_aid, f"Game #{best_h_aid}")

            explanation = (
                f"Recommended based on your play of '{best_title}' "
                f"(Item co-occurrence similarity: {contribs[best_pos]:.3f})."
            )

            recommendations.append({
                "app_id": aid,
                "title": title,
                "rank": rank,
                "score": round(score, 4),
                "model": "Item-Item Collaborative",
                "explanation": explanation,
            })

        return recommendations

    def save(self, file_path: Union[str, Path]) -> None:
        """Persists model to disk."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Saved ItemItemCollaborativeRecommender to {file_path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ItemItemCollaborativeRecommender":
        """Loads model from disk."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found at {file_path}")
        return joblib.load(file_path)


class MatrixFactorizationRecommender:
    """
    Latent Factor Matrix Factorization Recommender.
    Decomposes the confidence-weighted user-item interaction matrix using TruncatedSVD.
    
    Formulation:
    R_norm approx P @ Q.T
    where:
      - P in R^{|U| x d} (User latent representations)
      - Q in R^{|I| x d} (Item latent representations, Q = V)
    
    Score for user u:
    y_hat_u = p_u @ Q.T = (r_u @ Q) @ Q.T
    """

    def __init__(
        self,
        n_components: int = 64,
        random_state: int = 42,
        alpha_hours: float = 0.5,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.alpha_hours = alpha_hours
        self.is_fitted = False

        self.svd: Optional[TruncatedSVD] = None
        self.item_factors: Optional[np.ndarray] = None  # (n_items, d)
        self.user_to_idx: Dict[int, int] = {}
        self.app_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.idx_to_app: Dict[int, int] = {}
        self.user_history: Dict[int, Set[int]] = {}
        self.catalog_titles: Dict[int, str] = {}
        self.interaction_matrix: Optional[csr_matrix] = None

    def fit(
        self,
        train_interactions: pl.DataFrame,
        catalog_df: Optional[pl.DataFrame] = None,
    ) -> "MatrixFactorizationRecommender":
        """
        Fits TruncatedSVD latent factors strictly using training data.
        """
        logger.info(
            f"Fitting MatrixFactorizationRecommender ({self.n_components} components, seed={self.random_state})..."
        )

        if catalog_df is not None:
            self.catalog_titles = dict(
                zip(catalog_df["app_id"].to_list(), catalog_df["title"].to_list())
            )

        R, u2i, a2i, i2u, i2a = build_interaction_matrix(
            train_interactions,
            alpha_hours=self.alpha_hours,
            filter_positive_only=True,
        )
        self.interaction_matrix = R
        self.user_to_idx = u2i
        self.app_to_idx = a2i
        self.idx_to_user = i2u
        self.idx_to_app = i2a

        # Store training history
        pos_train = train_interactions.filter(pl.col("is_recommended") == True)
        self.user_history = {}
        for row in pos_train.select(["user_id", "app_id"]).iter_rows():
            uid, aid = int(row[0]), int(row[1])
            if uid not in self.user_history:
                self.user_history[uid] = set()
            self.user_history[uid].add(aid)

        # Row-normalize interaction matrix (cosine normalization per user)
        R_norm = normalize(R, norm="l2", axis=1)

        # Fit TruncatedSVD
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.svd.fit(R_norm)

        # Item factors: V.T -> Q = svd.components_.T (n_items, d)
        self.item_factors = self.svd.components_.T.astype(np.float32)
        # L2-normalize item factors for stable cosine-like dot products
        self.item_factors = normalize(self.item_factors, norm="l2", axis=1)

        logger.info(
            f"Matrix Factorization fit complete. Explained variance ratio sum: "
            f"{float(self.svd.explained_variance_ratio_.sum()):.4f}."
        )
        self.is_fitted = True
        return self

    def recommend_user(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: Optional[Union[Set[int], List[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-K items for a known user via latent factor projection.
        """
        if not self.is_fitted or self.item_factors is None or self.interaction_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in training matrix. Insufficient history.")
            return []

        user_idx = self.user_to_idx[user_id]
        user_vec = self.interaction_matrix[user_idx]
        user_vec_norm = normalize(user_vec, norm="l2", axis=1)

        # Project user into latent space: p_u = r_u @ Q
        user_latent = (user_vec_norm @ self.item_factors).flatten()
        user_latent_norm = user_latent / (np.linalg.norm(user_latent) + 1e-8)

        # Predicted affinities across all items: s = p_u @ Q.T
        scores = self.item_factors @ user_latent_norm

        # Exclude seen items
        seen_set = self.user_history.get(user_id, set()).copy()
        if exclude_seen:
            seen_set.update(exclude_seen)

        for aid in seen_set:
            if aid in self.app_to_idx:
                scores[self.app_to_idx[aid]] = -1.0

        top_indices = np.argsort(scores)[::-1][:k]

        recommendations = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                break
            aid = int(self.idx_to_app[idx])
            title = self.catalog_titles.get(aid, f"Game #{aid}")

            explanation = (
                f"Matches your latent collaborative gaming taste profile "
                f"(Latent affinity: {score:.3f})."
            )

            recommendations.append({
                "app_id": aid,
                "title": title,
                "rank": rank,
                "score": round(score, 4),
                "model": "Matrix Factorization (SVD)",
                "explanation": explanation,
            })

        return recommendations

    def recommend_items(
        self,
        history_app_ids: List[int],
        k: int = 10,
        exclude_seen: Optional[Union[Set[int], List[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-K items on-the-fly for any list of played games via latent projection.
        """
        if not self.is_fitted or self.item_factors is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        valid_indices = [self.app_to_idx[a] for a in history_app_ids if a in self.app_to_idx]
        if not valid_indices:
            return []

        # Average latent vectors of historical games
        user_latent = self.item_factors[valid_indices].mean(axis=0)
        user_latent_norm = user_latent / (np.linalg.norm(user_latent) + 1e-8)

        scores = self.item_factors @ user_latent_norm

        seen_set = set(history_app_ids)
        if exclude_seen:
            seen_set.update(exclude_seen)

        for aid in seen_set:
            if aid in self.app_to_idx:
                scores[self.app_to_idx[aid]] = -1.0

        top_indices = np.argsort(scores)[::-1][:k]

        recommendations = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                break
            aid = int(self.idx_to_app[idx])
            title = self.catalog_titles.get(aid, f"Game #{aid}")

            explanation = (
                f"Matches your latent collaborative gaming taste profile "
                f"(Latent affinity: {score:.3f})."
            )

            recommendations.append({
                "app_id": aid,
                "title": title,
                "rank": rank,
                "score": round(score, 4),
                "model": "Matrix Factorization (SVD)",
                "explanation": explanation,
            })

        return recommendations

    def save(self, file_path: Union[str, Path]) -> None:
        """Persists model to disk."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Saved MatrixFactorizationRecommender to {file_path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "MatrixFactorizationRecommender":
        """Loads model from disk."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found at {file_path}")
        return joblib.load(file_path)
