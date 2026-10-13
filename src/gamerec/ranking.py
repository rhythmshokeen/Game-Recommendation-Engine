"""
GameRec Ranking, Score Normalization & Diversity Module.
Provides score calibration, weighted hybrid combination, and Maximal Marginal Relevance (MMR)
diversity reranking for recommendation candidate lists.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger("gamerec.ranking")


def min_max_scale(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Linearly scales an array of numeric scores to the [0.0, 1.0] interval.
    Safely handles constant arrays and NaN/Inf values.
    """
    if len(scores) == 0:
        return scores

    valid_mask = np.isfinite(scores) & (scores > -1e8)
    if not np.any(valid_mask):
        return np.zeros_like(scores, dtype=np.float32)

    valid_vals = scores[valid_mask]
    s_min = float(np.min(valid_vals))
    s_max = float(np.max(valid_vals))

    if abs(s_max - s_min) < eps:
        # Constant array
        scaled = np.zeros_like(scores, dtype=np.float32)
        scaled[valid_mask] = 1.0
        return scaled

    scaled = np.zeros_like(scores, dtype=np.float32)
    scaled[valid_mask] = (scores[valid_mask] - s_min) / (s_max - s_min + eps)
    return np.clip(scaled, 0.0, 1.0)


def extract_franchise_key(title: str) -> str:
    """
    Extracts an approximate franchise key from a game title for diversity deduplication.
    Examples:
      'The Witcher 3: Wild Hunt' -> 'the witcher'
      'Tomb Raider: Underworld' -> 'tomb raider'
      'Half-Life 2: Episode One' -> 'half life'
    """
    clean = re.sub(r"[^\w\s]", " ", title.lower()).strip()
    words = clean.split()
    if not words:
        return title.lower()

    # Common franchise patterns
    if words[0] in ["the", "a"] and len(words) > 1:
        base = " ".join(words[:2])
    else:
        base = words[0]

    # Handle multi-word franchise names
    multiword_known = [
        "tomb raider", "half life", "counter strike", "mount blade",
        "borderlands", "witcher", "portal", "dark souls", "fallout",
        "elder scrolls", "civilization", "total war", "assassin creed",
    ]
    for franch in multiword_known:
        if franch in clean:
            return franch

    return base


def combine_hybrid_scores(
    candidate_app_ids: List[int],
    content_scores_dict: Dict[int, float],
    collab_scores_dict: Dict[int, float],
    alpha: float = 0.5,
) -> Dict[int, Dict[str, float]]:
    """
    Combines normalized Content and Collaborative scores with weighting parameter alpha.
    
    alpha = 1.0 -> 100% Content-Based
    alpha = 0.0 -> 100% Collaborative Filtering
    0.0 < alpha < 1.0 -> Weighted Linear Blend
    
    Returns:
        Dict mapping app_id -> {'hybrid_score': float, 'content_score': float, 'collab_score': float}
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be between 0.0 and 1.0; received {alpha}")

    results = {}
    for aid in candidate_app_ids:
        c_score = content_scores_dict.get(aid, 0.0)
        cf_score = collab_scores_dict.get(aid, 0.0)

        # Handle alpha boundary cases strictly
        if alpha == 1.0:
            h_score = c_score
        elif alpha == 0.0:
            h_score = cf_score
        else:
            # Weighted combination
            # If item has no collaborative score (cold-start item), rely on content
            if aid not in collab_scores_dict:
                h_score = alpha * c_score
            else:
                h_score = (alpha * c_score) + ((1.0 - alpha) * cf_score)

        results[aid] = {
            "hybrid_score": float(np.round(h_score, 4)),
            "content_score": float(np.round(c_score, 4)),
            "collab_score": float(np.round(cf_score, 4)),
        }

    return results


def maximal_marginal_relevance(
    candidates: List[Dict[str, Any]],
    feature_matrix: Optional[csr_matrix] = None,
    app_to_idx: Optional[Dict[int, int]] = None,
    diversity_lambda: float = 0.7,
    top_k: int = 10,
    max_per_franchise: int = 2,
) -> List[Dict[str, Any]]:
    """
    Reranks candidates using Maximal Marginal Relevance (MMR) and franchise caps
    to suppress near-duplicate recommendations and ensure catalog diversity.
    
    diversity_lambda = 1.0: Pure relevance (no diversity penalty).
    diversity_lambda = 0.0: Maximal diversity.
    Default 0.7: High relevance with suppression of redundant sequels/DLCs.
    """
    if not candidates or top_k <= 0:
        return []

    if top_k >= len(candidates) and max_per_franchise <= 0:
        return candidates[:top_k]

    selected: List[Dict[str, Any]] = []
    selected_indices: List[int] = []
    selected_app_ids: Set[int] = set()
    franchise_counts: Dict[str, int] = {}

    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        best_candidate = None
        best_mmr_score = -float("inf")
        best_cand_pos = -1

        for pos, cand in enumerate(remaining):
            aid = cand["app_id"]
            title = cand["title"]
            rel_score = cand.get("score", cand.get("hybrid_score", 0.0))

            # Check franchise cap
            if max_per_franchise > 0:
                franch_key = extract_franchise_key(title)
                if franchise_counts.get(franch_key, 0) >= max_per_franchise:
                    continue

            # Compute redundancy penalty with already selected items
            if not selected:
                redundancy = 0.0
            else:
                # If feature matrix is available, compute max cosine similarity with selected items
                if feature_matrix is not None and app_to_idx is not None and aid in app_to_idx:
                    c_idx = app_to_idx[aid]
                    cand_vec = feature_matrix[c_idx]
                    sims = [
                        float((cand_vec @ feature_matrix[s_idx].T).toarray()[0, 0])
                        for s_idx in selected_indices
                    ]
                    redundancy = max(sims) if sims else 0.0
                else:
                    # Fallback to shared tags overlap penalty
                    cand_tags = set(cand.get("all_tags", cand.get("tags", [])))
                    if cand_tags:
                        overlaps = [
                            len(cand_tags.intersection(set(s.get("all_tags", s.get("tags", []))))) / len(cand_tags)
                            for s in selected
                        ]
                        redundancy = max(overlaps) if overlaps else 0.0
                    else:
                        redundancy = 0.0

            # MMR Objective: lambda * Relevance - (1 - lambda) * Redundancy
            mmr_score = (diversity_lambda * rel_score) - ((1.0 - diversity_lambda) * redundancy)

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate = cand
                best_cand_pos = pos

        if best_candidate is None:
            # All remaining candidates exceed franchise cap; break or loosen cap
            break

        # Append best candidate to selected
        selected.append(best_candidate)
        selected_app_ids.add(best_candidate["app_id"])
        if app_to_idx is not None and best_candidate["app_id"] in app_to_idx:
            selected_indices.append(app_to_idx[best_candidate["app_id"]])

        if max_per_franchise > 0:
            franch_key = extract_franchise_key(best_candidate["title"])
            franchise_counts[franch_key] = franchise_counts.get(franch_key, 0) + 1

        remaining.pop(best_cand_pos)

    # Re-assign sequential ranks
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank

    return selected
