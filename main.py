"""Main CLI entrypoint to execute the Cellulose Bleaching Linear Regression Pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CONFIG
from src.pipeline import IndustrialBleachingPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Industrial Cellulose Bleaching Process - Linear Regression Analytics Suite"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-reading the raw Excel file instead of cached Parquet."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating PNG visual figures in reports/figures/."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = IndustrialBleachingPipeline(config=CONFIG)
    result = pipeline.run(
        use_cache=not args.no_cache,
        generate_plots=not args.no_plots
    )
    print(f"\n[OK] Pipeline completed successfully. Best model: {result.best_model_name}")


if __name__ == "__main__":
    main()
