"""Main pipeline entry point. Invoke with spark-submit.

Example:
    spark-submit --master local[4] scripts/run_pipeline.py \\
        --config configs/local.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> int:
    # Make the package importable when run via spark-submit, which doesn't
    # always honor pip-installed packages on workers.
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from wmh_spark.config import load_config
    from wmh_spark.pipeline import run_pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
