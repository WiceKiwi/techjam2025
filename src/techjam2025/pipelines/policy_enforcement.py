from __future__ import annotations
import json, os
from typing import Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class PolicyEnforcementPipeline:
    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg
        self.logger = logger
        self._validate()

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    # ---------------- IO helpers ----------------
    def _read_any(self, path: str) -> pd.DataFrame:
        if path.endswith(".jsonl"): return pd.read_json(path, lines=True)
        if path.endswith(".json"):  return pd.read_json(path)
        if path.endswith(".parquet"): return pd.read_parquet(path)
        if path.endswith(".csv"):   return pd.read_csv(path)
        raise ValueError(f"unsupported format: {path}")

    def _write_jsonl(self, df: pd.DataFrame, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = df.where(pd.notnull(df), None)
        with open(path, "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --------------- validation -----------------
    def _validate(self):
        need = ["data","schema","thresholds","genuine","output","logging"]
        for k in need:
            if k not in self.cfg: raise ValueError(f"missing config: {k}")
        if "id_key" not in self.cfg["schema"]:
            raise ValueError("schema.id_key is required")
        th = self.cfg["thresholds"]
        if "labels" not in th or "continuous" not in th:
            raise ValueError("thresholds.labels and thresholds.continuous required")

    # --------------- genuine score --------------
    def _genuine_score(self, row, method="geom_mean") -> float:
        # Use predictions; fall back to 0 if missing.
        bad = [
            1.0 - float(row.get("pred_spam_low_quality", 0.0) or 0.0),
            1.0 - float(row.get("pred_ads_promo",        0.0) or 0.0),
            1.0 - float(row.get("pred_irrelevant",       0.0) or 0.0),
            1.0 - float(row.get("pred_rant_no_visit",    0.0) or 0.0),
        ]
        good = [
            float(row.get("pred_relevancy_score",  0.0) or 0.0),
            float(row.get("pred_visit_likelihood", 0.0) or 0.0),
        ]
        comps = np.clip(np.array(bad + good, dtype=float), 0.0, 1.0)
        if comps.size == 0:
            return 0.0
        if method == "min":
            return float(comps.min())
        # geom mean
        comps = np.where(comps <= 0, 1e-6, comps)
        return float(np.exp(np.log(comps).mean()))

    # --------------- decision logic -------------
    def _decide(self, row: dict) -> dict:
        th = self.cfg["thresholds"]
        labels = th["labels"]
        cont   = th["continuous"]

        action = "allow"      # default; escalate to review/remove
        reasons: List[str] = []
        states  = {}

        # BAD labels (remove/review gates)
        for lab, pair in labels.items():
            p = float(row.get(f"pred_{lab}", 0.0) or 0.0)
            states[f"p_{lab}"] = p
            auto = float(pair.get("auto", 1.0))
            rev  = float(pair.get("review", 1.0))
            if p >= auto:
                action = "remove"
                reasons.append(f"{lab}>={auto:.2f}")
            elif p >= rev and action != "remove":
                action = "review"
                reasons.append(f"{lab}>={rev:.2f}")

        # Continuous “low” gates
        def _gate_low(name: str, auto_max: float, review_max: float):
            nonlocal action, reasons
            p = float(row.get(f"pred_{name}", 0.0) or 0.0)
            states[f"p_{name}"] = p
            if p <= auto_max:
                action = "remove"
                reasons.append(f"{name}<={auto_max:.2f}")
            elif p <= review_max and action != "remove":
                action = "review"
                reasons.append(f"{name}<={review_max:.2f}")

        _gate_low("visit_likelihood",
                  float(cont["visit_likelihood"]["auto_max"]),
                  float(cont["visit_likelihood"]["review_max"]))
        _gate_low("relevancy_score",
                  float(cont["relevancy_score"]["auto_max"]),
                  float(cont["relevancy_score"]["review_max"]))

        # Genuine score & final label
        G = self.cfg["genuine"]
        gscore = self._genuine_score(row, method=G.get("score_method","geom_mean"))
        auto_min   = float(G.get("auto_min", 0.85))
        review_min = float(G.get("review_min", 0.65))

        if action == "remove":
            final = "not_genuine"
        elif action == "review":
            final = "review"
        else:
            # action == allow → consider gscore
            if gscore >= auto_min:
                final = "genuine"
                reasons.append(f"genuine_score>={auto_min:.2f}")
            elif gscore >= review_min:
                final = "review"
                reasons.append(f"genuine_score>={review_min:.2f}")
            else:
                final = "not_genuine"
                reasons.append(f"genuine_score<{review_min:.2f}")

        return {
            "action": action,
            "final": final,
            "reasons": reasons,
            "states": states,
            "genuine_score": gscore,
        }

    # --------------- run for a split/file -------
    def _run_one(self, base_path: str, preds_path: str) -> pd.DataFrame:
        id_key = self.cfg["schema"]["id_key"]
        carry  = self.cfg.get("carry_columns", [])
        use_tqdm = self.cfg.get("logging", {}).get("use_tqdm", True) and tqdm is not None

        base  = self._read_any(base_path)
        preds = self._read_any(preds_path)

        # Required prediction columns
        needed = [
            "pred_ads_promo", "pred_spam_low_quality", "pred_irrelevant",
            "pred_rant_no_visit", "pred_relevancy_score", "pred_visit_likelihood"
        ]
        missing = [c for c in [id_key] + needed if c not in preds.columns]
        if missing:
            raise ValueError(f"preds file missing columns: {missing}")

        merged = base[[id_key] + carry].merge(preds[[id_key] + needed], on=id_key, how="left")
        self._log(f"merged_rows={len(merged)} from base={len(base)} preds={len(preds)}")

        rows = []
        iterator = tqdm(merged.to_dict(orient="records"), total=len(merged), disable=not use_tqdm)
        for rec in iterator:
            d = self._decide(rec)
            out = {
                id_key: rec.get(id_key, ""),
                "action": d["action"],
                "final": d["final"],
                "genuine_score": float(d["genuine_score"]),
                "reasons": d["reasons"],
            }
            # keep predictions and any carry columns for traceability
            for k, v in rec.items():
                if k == id_key: continue
                out[k] = v
            rows.append(out)

        return pd.DataFrame(rows)

    # --------------- public entry ----------------
    def run(self):
        outcfg = self.cfg["output"]
        Path(outcfg.get("dir", "artifacts/policy")).mkdir(parents=True, exist_ok=True)

        dcfg = self.cfg["data"]
        split = dcfg.get("split", "test")
        if split == "file":
            base_path  = dcfg["base_path"]
            preds_path = dcfg["preds_path"]
        else:
            base_path  = dcfg["base_paths"][split]
            preds_path = dcfg["preds_paths"][split]

        # Warnings
        if self.cfg.get("logging", {}).get("silence_warnings", False):
            import warnings
            warnings.filterwarnings("ignore")

        df = self._run_one(base_path, preds_path)

        # Save main decisions
        self._write_jsonl(df, outcfg["path"])
        self._log(f"exported_decisions={outcfg['path']} rows={len(df)}")

        # Simple aggregates
        counts = (
            df.assign(n=1)
              .groupby(["final","action"], dropna=False)["n"]
              .sum()
              .reset_index()
              .sort_values("n", ascending=False)
        )
        Path(outcfg["counts_csv"]).parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(outcfg["counts_csv"], index=False)

        # Exploded reasons for audit
        rx = df[[self.cfg["schema"]["id_key"], "final", "action", "reasons"]].explode("reasons")
        rx.to_csv(outcfg["reasons_csv"], index=False)

        self._log(f"counts -> {outcfg['counts_csv']}")
        self._log(f"reasons -> {outcfg['reasons_csv']}")
