#!/usr/bin/env python3
"""
GameRec Preprocessing Runner.
Executes the data cleaning, deduplication, catalog unification,
and benchmark extraction pipeline.
"""

import sys
import time
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gamerec.preprocessing import run_preprocessing_pipeline

def main():
    print("=" * 70)
    print("GameRec: Running Phase 2 Preprocessing Pipeline")
    print("=" * 70)
    t0 = time.time()
    
    results = run_preprocessing_pipeline()
    
    elapsed = round(time.time() - t0, 2)
    print("\n" + "=" * 70)
    print(f"Pipeline completed successfully in {elapsed}s!")
    print(f"Catalog output:   {results['catalog_path']}")
    print(f"Benchmark output: {results['benchmark_path']}")
    print(f"Metrics output:   {results['metrics_path']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
