"""
GameRec Offline Ranking Evaluation Module.
Computes standard recommendation and ranking metrics:
- Hit Rate@K (HR@K)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Precision@K
- Mean Average Precision (MAP@K)
- Mean Reciprocal Rank (MRR)
strictly on held-out temporal evaluation test sets with zero data leakage.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np
import polars as pl

logger = logging.getLogger("gamerec.evaluation")


def compute_user_metrics(
    recommended_ids: List[int],
    ground_truth_id: int,
    ks: Tuple[int, ...] = (5, 10, 20),
) -> Dict[str, float]:
    """
    Computes ranking metrics for a single user given recommended items and target item.
    Assumes leave-last-out evaluation (1 ground-truth relevant item per user).
    
    Args:
        recommended_ids: Ranked list of recommended app_ids (from rank 1 to max(ks)).
        ground_truth_id: The true held-out positive game app_id.
        ks: Cutoff values for evaluation metrics.
        
    Returns:
        Dict containing hit@k, ndcg@k, precision@k, map@k for each k, plus mrr.
    """
    metrics: Dict[str, float] = {}

    # Find position of ground-truth item (1-indexed) if present
    rank: Optional[int] = None
    for idx, aid in enumerate(recommended_ids, start=1):
        if aid == ground_truth_id:
            rank = idx
            break

    # MRR (over entire recommendation list)
    metrics["mrr"] = (1.0 / rank) if rank is not None else 0.0

    for k in ks:
        if rank is not None and rank <= k:
            # Item found within top K
            hit = 1.0
            ndcg = 1.0 / math.log2(rank + 1.0)
            precision = 1.0 / float(k)
            map_score = 1.0 / float(rank)
        else:
            hit = 0.0
            ndcg = 0.0
            precision = 0.0
            map_score = 0.0

        metrics[f"hit@{k}"] = hit
        metrics[f"ndcg@{k}"] = ndcg
        metrics[f"precision@{k}"] = precision
        metrics[f"map@{k}"] = map_score

    return metrics


def aggregate_metrics(all_user_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Computes mean across all evaluated users for each metric."""
    if not all_user_metrics:
        return {}

    metric_names = list(all_user_metrics[0].keys())
    aggregated = {}
    n = len(all_user_metrics)

    for name in metric_names:
        values = [m[name] for m in all_user_metrics]
        aggregated[name] = float(np.mean(values))

    return aggregated


def format_metrics_table(results_dict: Dict[str, Dict[str, float]]) -> str:
    """Formats evaluation results as a clean Markdown table."""
    headers = [
        "Model",
        "Hit@5 (%)", "Hit@10 (%)", "Hit@20 (%)",
        "NDCG@10", "NDCG@20",
        "Precision@10", "MAP@10", "MRR",
    ]
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for model_name, m in results_dict.items():
        row = [
            f"**{model_name}**",
            f"{m.get('hit@5', 0.0) * 100:.2f}%",
            f"{m.get('hit@10', 0.0) * 100:.2f}%",
            f"{m.get('hit@20', 0.0) * 100:.2f}%",
            f"{m.get('ndcg@10', 0.0):.4f}",
            f"{m.get('ndcg@20', 0.0):.4f}",
            f"{m.get('precision@10', 0.0):.4f}",
            f"{m.get('map@10', 0.0):.4f}",
            f"{m.get('mrr', 0.0):.4f}",
        ]
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)
