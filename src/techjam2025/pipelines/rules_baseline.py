from __future__ import annotations
from typing import Dict, Any, List
import json
import pandas as pd
from pathlib import Path

class RulesBaselinePipeline:
    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg
        self.logger = logger
        self._validate_cfg()

    def _log(self, msg: str):
        if self.logger: self.logger.info(msg)

    def _validate_cfg(self):
        req = ["input_path","output_path","thresholds","signals"]
        for k in req:
            if k not in self.cfg: raise ValueError(f"missing config: {k}")
        for k in ["ads","spam","irrelevant","rant_no_visit"]:
            if k not in self.cfg["thresholds"]: raise ValueError(f"missing thresholds.{k}")
        for k in ["emoji_count_min","punct_runs_min","elong_runs_min","duplicate_text_min"]:
            if k not in self.cfg["signals"]: raise ValueError(f"missing signals.{k}")

    def _read_input(self) -> pd.DataFrame:
        p = self.cfg["input_path"]
        if p.endswith(".jsonl"): return pd.read_json(p, lines=True)
        if p.endswith(".json"): return pd.read_json(p)
        if p.endswith(".parquet"): return pd.read_parquet(p)
        if p.endswith(".csv"): return pd.read_csv(p)
        raise ValueError(f"unsupported input format: {p}")

    def _export(self, df: pd.DataFrame):
        out = self.cfg["output_path"]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if out.endswith(".jsonl"):
            with open(out, "w", encoding="utf-8") as f:
                for rec in df.to_dict(orient="records"):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        elif out.endswith(".json"):
            df.to_json(out, orient="records", force_ascii=False)
        elif out.endswith(".parquet"):
            df.to_parquet(out, index=False)
        elif out.endswith(".csv"):
            df.to_csv(out, index=False)
        else:
            raise ValueError(f"unsupported output format: {out}")
        self._log(f"exported_rules={out}")

    @staticmethod
    def _col(df: pd.DataFrame, name: str, default=False):
        return df[name] if name in df.columns else default

    def _ads_block(self, f: pd.DataFrame) -> pd.DataFrame:
        # boolean signals
        cols = [c for c in [
            "has_url","has_domain","has_short_url","has_phone_any","has_phone_us","has_email",
            "has_currency","has_percent_off","has_promo_code","has_cta","has_social_handle","has_hashtag"
        ] if c in f.columns]
        ads_bool_sum = f[cols].astype(int).sum(axis=1) if cols else 0
        # counts from lexicons (clip to 1 so they behave like boolean hints)
        promo_hits = f["promo_hits"].clip(0,1) if "promo_hits" in f.columns else 0
        contact_hits = f["contact_hits"].clip(0,1) if "contact_hits" in f.columns else 0
        denom = (len(cols) + 2) or 1
        score = (ads_bool_sum + promo_hits + contact_hits) / denom
        return pd.DataFrame({
            "rule_ads_score": score,
        }, index=f.index)

    def _spam_block(self, f: pd.DataFrame) -> pd.DataFrame:
        sig = self.cfg["signals"]
        short = (f["length_bucket"] == "short") if "length_bucket" in f.columns else False
        has_short_url = self._col(f, "has_short_url", False)
        punct = (self._col(f, "punct_runs", 0) >= int(sig["punct_runs_min"]))
        elong = (self._col(f, "elong_runs", 0) >= int(sig["elong_runs_min"]))
        emoji = (self._col(f, "emoji_count", 0) >= int(sig["emoji_count_min"]))
        dup_place = (self._col(f, "text_dup_count_place", 1) >= int(sig["duplicate_text_min"]))
        mat = pd.concat([
            short.astype(int) if hasattr(short, "astype") else pd.Series(0, index=f.index),
            has_short_url.astype(int) if hasattr(has_short_url, "astype") else pd.Series(0, index=f.index),
            punct.astype(int) if hasattr(punct, "astype") else pd.Series(0, index=f.index),
            elong.astype(int) if hasattr(elong, "astype") else pd.Series(0, index=f.index),
            emoji.astype(int) if hasattr(emoji, "astype") else pd.Series(0, index=f.index),
            dup_place.astype(int) if hasattr(dup_place, "astype") else pd.Series(0, index=f.index),
        ], axis=1)
        score = mat.sum(axis=1) / mat.shape[1]
        return pd.DataFrame({"rule_spam_score": score}, index=f.index)

    def _irrelevant_block(self, f: pd.DataFrame) -> pd.DataFrame:
        offtopic = (self._col(f, "offtopic_hits", 0) > 0)
        mentions_cat = None
        for c in ["mentions_category","mentions_categories","mentions_types"]:
            if c in f.columns:
                mentions_cat = f[c].astype(bool); break
        if mentions_cat is None:
            score = offtopic.astype(int)
        else:
            score = (offtopic & (~mentions_cat)).astype(int)
        return pd.DataFrame({"rule_irrelevant_score": score}, index=f.index)

    def _rant_block(self, f: pd.DataFrame) -> pd.DataFrame:
        score = (self._col(f, "nonvisit_hits", 0) > 0).astype(int)
        return pd.DataFrame({"rule_rant_score": score}, index=f.index)

    def compute(self, feats: pd.DataFrame) -> pd.DataFrame:
        thr = self.cfg["thresholds"]
        out = feats.copy()
        # compute scores
        ads = self._ads_block(out)
        spam = self._spam_block(out)
        irr = self._irrelevant_block(out)
        rant = self._rant_block(out)
        out = pd.concat([out, ads, spam, irr, rant], axis=1)

        # strong flags
        out["rule_ads_strong"] = (out["rule_ads_score"] >= float(thr["ads"])).astype(int)
        out["rule_spam_strong"] = (out["rule_spam_score"] >= float(thr["spam"])).astype(int)
        out["rule_irrelevant_strong"] = (out["rule_irrelevant_score"] >= float(thr["irrelevant"])).astype(int)
        out["rule_rant_strong"] = (out["rule_rant_score"] >= float(thr["rant_no_visit"])).astype(int)

        present = ["rule_ads_strong","rule_spam_strong","rule_irrelevant_strong","rule_rant_strong"]
        out["rule_none"] = (out[present].sum(axis=1) == 0).astype(int)

        self._log(f"rules_rows={len(out)} cols_added=9")
        return out

    def run(self):
        feats = self._read_input()
        self._log(f"loaded_features_rows={len(feats)}")
        out = self.compute(feats)
        self._export(out)
        return out
