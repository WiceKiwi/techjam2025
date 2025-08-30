import os, sys, yaml, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from techjam2025.pipelines.dataset_split import DatasetSplitPipeline

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def build_logger(level="INFO"):
    logger = logging.getLogger("dataset-split")
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [h]
    return logger

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config","-c",default=str(ROOT/"configs"/"dataset_split.yaml"))
    ap.add_argument("--log-level",default="INFO")
    args = ap.parse_args()

    logger = build_logger(args.log_level)
    cfg = load_yaml(args.config)
    pipe = DatasetSplitPipeline(cfg, logger=logger)
    pipe.run()

if __name__ == "__main__":
    main()
