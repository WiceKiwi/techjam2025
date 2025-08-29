import os, sys, yaml, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from techjam2025.pipelines.feature_engineering import FeatureEngineeringPipeline

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    def expand(x):
        if isinstance(x, str): return os.path.expandvars(x)
        if isinstance(x, list): return [expand(v) for v in x]
        if isinstance(x, dict): return {k: expand(v) for k, v in x.items()}
        return x
    return expand(cfg)

def build_logger(level="INFO"):
    logger = logging.getLogger("features")
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [handler]
    return logger

def validate_cfg(cfg):
    required = ["input_path","output_path","enable_groups","length","unicode","lexicons"]
    for k in required:
        if k not in cfg: raise ValueError(f"missing config: {k}")
    for k in ["short_tokens_max","long_chars_min"]:
        if k not in cfg["length"]: raise ValueError(f"missing length.{k}")
    if "nonlatin_ratio_flag" not in cfg["unicode"]:
        raise ValueError("missing unicode.nonlatin_ratio_flag")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config","-c",default=str(ROOT/"configs"/"feature_engineering.yaml"))
    ap.add_argument("--log-level",default="INFO")
    args = ap.parse_args()

    logger = build_logger(args.log_level)
    cfg = load_yaml(args.config)
    validate_cfg(cfg)

    pipe = FeatureEngineeringPipeline(cfg, logger=logger)
    pipe.run()

if __name__ == "__main__":
    main()
