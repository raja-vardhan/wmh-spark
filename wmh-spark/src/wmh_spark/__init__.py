"""wmh_spark: scalable WMH quantification on Apache Spark.

Public API
==========
- ``run_pipeline(cfg)``  -- end-to-end pipeline
- ``load_config(path)``  -- load a YAML config into a frozen dataclass
- ``Config``             -- top-level config schema
"""

from .config import Config, load_config
from .pipeline import run_pipeline

__all__ = ["Config", "load_config", "run_pipeline"]
__version__ = "0.1.0"
