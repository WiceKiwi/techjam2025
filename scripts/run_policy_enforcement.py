import argparse, json, logging, sys
from pathlib import Path
import yaml

# Ensure src on path if needed
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from techjam2025.pipelines.policy_enforcement import PolicyEnforcementPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("policy")

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    pipe = PolicyEnforcementPipeline(cfg, logger=logger)
    pipe.run()


if __name__ == "__main__":
    main()
