import os, sys, json, yaml, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam2025.pipelines.model_training import ModelTrainingPipeline

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def build_logger(level="INFO"):
    logger = logging.getLogger("model-training")
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [h]
    return logger

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default=str(ROOT / "configs" / "model_training.yaml"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = build_logger(args.log_level)
    cfg = load_yaml(args.config)

    pipe = ModelTrainingPipeline(cfg, logger=logger)
    pipe.run()

if __name__ == "__main__":
    main()
