"""
GameRec Data Ingestion Module.
Provides safe, memory-efficient data loading routines for Steam raw and processed datasets.
"""

import json
import os
from pathlib import Path
from typing import Generator, Optional, Union, Dict, Any
import polars as pl
import pandas as pd


def get_default_paths(project_root: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """Returns standardized paths for raw and processed datasets."""
    if project_root is None:
        # Default to repository root
        project_root = Path(__file__).resolve().parent.parent.parent
    else:
        project_root = Path(project_root)

    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"

    return {
        "project_root": project_root,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "reports_dir": reports_dir,
        "raw_games": raw_dir / "games.csv",
        "raw_metadata": raw_dir / "games_metadata.json",
        "raw_recommendations": raw_dir / "recommendations.csv",
        "raw_users": raw_dir / "users.csv",
        "processed_catalog": processed_dir / "games_catalog.parquet",
        "processed_interactions": processed_dir / "interactions_clean.parquet",
        "processed_benchmark": processed_dir / "interactions_benchmark.parquet",
    }


def load_raw_games(path: Optional[Union[str, Path]] = None) -> pl.DataFrame:
    """Loads raw games.csv into a Polars DataFrame with type assertions."""
    if path is None:
        path = get_default_paths()["raw_games"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Games dataset not found at {path}")

    df = pl.read_csv(
        path,
        schema_overrides={
            "app_id": pl.Int64,
            "title": pl.String,
            "date_release": pl.String,
            "win": pl.Boolean,
            "mac": pl.Boolean,
            "linux": pl.Boolean,
            "rating": pl.String,
            "positive_ratio": pl.Int64,
            "user_reviews": pl.Int64,
            "price_final": pl.Float64,
            "price_original": pl.Float64,
            "discount": pl.Float64,
            "steam_deck": pl.Boolean,
        },
    )
    return df


def load_raw_metadata(path: Optional[Union[str, Path]] = None) -> pl.DataFrame:
    """
    Loads raw games_metadata.json (JSON Lines format) memory-efficiently into a Polars DataFrame.
    """
    if path is None:
        path = get_default_paths()["raw_metadata"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Games metadata not found at {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pl.DataFrame(
        records,
        schema={
            "app_id": pl.Int64,
            "description": pl.String,
            "tags": pl.List(pl.String),
        },
    )
    return df


def load_raw_recommendations_lazy(
    path: Optional[Union[str, Path]] = None,
) -> pl.LazyFrame:
    """
    Returns a Polars LazyFrame scanning recommendations.csv without loading into RAM.
    Allows zero-copy predicate pushdown and chunked streaming.
    """
    if path is None:
        path = get_default_paths()["raw_recommendations"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recommendations dataset not found at {path}")

    return pl.scan_csv(
        path,
        schema_overrides={
            "app_id": pl.Int64,
            "helpful": pl.Int64,
            "funny": pl.Int64,
            "date": pl.String,
            "is_recommended": pl.Boolean,
            "hours": pl.Float64,
            "user_id": pl.Int64,
            "review_id": pl.Int64,
        },
    )


def load_raw_users_lazy(path: Optional[Union[str, Path]] = None) -> pl.LazyFrame:
    """Returns a Polars LazyFrame scanning users.csv."""
    if path is None:
        path = get_default_paths()["raw_users"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Users dataset not found at {path}")

    return pl.scan_csv(
        path,
        schema_overrides={
            "user_id": pl.Int64,
            "products": pl.Int64,
            "reviews": pl.Int64,
        },
    )


def stream_recommendations_chunks(
    path: Optional[Union[str, Path]] = None,
    chunk_size: int = 200_000,
) -> Generator[pd.DataFrame, None, None]:
    """
    Generator yielding Pandas DataFrame chunks from recommendations.csv.
    Ensures safe reading without exhausting host RAM.
    """
    if path is None:
        path = get_default_paths()["raw_recommendations"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recommendations dataset not found at {path}")

    for chunk in pd.read_csv(
        path,
        chunksize=chunk_size,
        dtype={
            "app_id": "int64",
            "helpful": "int64",
            "funny": "int64",
            "date": "str",
            "is_recommended": "bool",
            "hours": "float64",
            "user_id": "int64",
            "review_id": "int64",
        },
    ):
        yield chunk


def load_processed_catalog(path: Optional[Union[str, Path]] = None) -> pl.DataFrame:
    """Loads clean processed games catalog from Parquet."""
    if path is None:
        path = get_default_paths()["processed_catalog"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed games catalog not found at {path}")

    return pl.read_parquet(path)


def load_processed_interactions(
    path: Optional[Union[str, Path]] = None,
    lazy: bool = True,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Loads clean processed interactions from Parquet."""
    if path is None:
        path = get_default_paths()["processed_benchmark"]
        if not path.exists():
            path = get_default_paths()["processed_interactions"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed interactions not found at {path}")

    if lazy:
        return pl.scan_parquet(path)
    return pl.read_parquet(path)
