import os, sys, yaml, glob, logging, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from src.techjam2025.pipelines.data_collection import DataCollectionPipeline, DataCollectionConfig

def load_yaml(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    def expand(x):
        if isinstance(x, str): return os.path.expandvars(x)
        if isinstance(x, list): return [expand(v) for v in x]
        if isinstance(x, dict): return {k: expand(v) for k, v in x.items()}
        return x
    return expand(cfg)

def build_logger(level="INFO"):
    logger = logging.getLogger("data-collection")
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [handler]
    return logger

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default=str(ROOT / "configs" / "data_collection.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="load + sample, but don't export")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = build_logger(args.log_level)
    cfg_dict = load_yaml(args.config)
    dc_cfg = DataCollectionConfig.from_dict(cfg_dict)
    pipe = DataCollectionPipeline(dc_cfg, logger=logger)

    # 1) load reviews + meta
    df_reviews, df_meta = pipe._load_sources()

    # 2) merge on gmap_id (or YAML meta.join_on)
    df_merged = DataCollectionPipeline.merge_metadata(
        df_reviews,
        df_meta,
        on=dc_cfg.meta.join_on,
        keep_cols=dc_cfg.meta.keep_cols,
        logger=logger
    )

    # 3) sample using YAML-driven knobs
    sample = DataCollectionPipeline.stratified_sample_reviews(df_merged, dc_cfg.sampler_cfg(), logger=logger)
    logger.info(f"sampled_rows={len(sample)}")

    # 4) export
    if not args.dry_run:
        pipe._export(sample)
        logger.info("export complete")
    else:
        logger.info("dry-run: export skipped")

if __name__ == "__main__":
    main()