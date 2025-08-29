import os, sys, yaml, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from techjam2025.pipelines.gold_labeling import GoldLabelingPipeline

def load_yaml(path):
    with open(path,"r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    def expand(x):
        if isinstance(x,str): return os.path.expandvars(x)
        if isinstance(x,list): return [expand(v) for v in x]
        if isinstance(x,dict): return {k:expand(v) for k,v in x.items()}
        return x
    return expand(cfg)

def build_logger(level="INFO"):
    logger = logging.getLogger("gold")
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [h]
    return logger

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config","-c",default=str(ROOT/"configs"/"gold_labeling.yaml"))
    ap.add_argument("--mode",choices=["build","ingest"],required=True)
    ap.add_argument("--log-level",default="INFO")
    args = ap.parse_args()

    logger = build_logger(args.log_level)
    cfg = load_yaml(args.config)
    pipe = GoldLabelingPipeline(cfg,logger=logger)

    if args.mode=="build":
        pipe.build_pack()
    else:
        pipe.ingest_labels()

if __name__=="__main__":
    main()
