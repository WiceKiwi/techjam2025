import os, sys, json, logging
from pathlib import Path

# add src to path (adjust if your layout differs)
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

import yaml

from techjam2025.pipelines.evaluation import EvaluationPipeline

def build_logger(level="INFO"):
    logger = logging.getLogger("eval")
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [h]
    return logger

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default=str(ROOT / "configs" / "evaluation.yaml"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = build_logger(args.log_level)
    cfg = load_yaml(args.config)

    # make sure output dirs exist
    for k in ("output",):
        if k in cfg and "dir" in cfg[k]:
            Path(cfg[k]["dir"]).mkdir(parents=True, exist_ok=True)
    if "output" in cfg and "plots_dir" in cfg["output"]:
        Path(cfg["output"]["plots_dir"]).mkdir(parents=True, exist_ok=True)

    pipe = EvaluationPipeline(cfg, logger=logger)
    pipe.run()

if __name__ == "__main__":
    main()
