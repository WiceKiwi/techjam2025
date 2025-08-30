from __future__ import annotations
from typing import Dict, Any, List, Optional
import json, math
import pandas as pd
import numpy as np

from techjam2025.utils.regexes import *

class FeatureEngineeringPipeline:
    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg
        self.logger = logger
        self._build_lexicons()

    def _log(self, msg: str):
        if self.logger: self.logger.info(msg)

    def _build_lexicons(self):
        lx = self.cfg.get("lexicons", {})
        self.re_promo = compile_lexicon_regex(lx.get("promo", []))
        self.re_contact = compile_lexicon_regex(lx.get("contact", []))
        self.re_nonvisit = compile_lexicon_regex(lx.get("nonvisit", []))
        self.re_offtopic = compile_lexicon_regex(lx.get("offtopic", []))

    def _read_input(self) -> pd.DataFrame:
        path = self.cfg["input_path"]
        if path.endswith(".jsonl"):
            return pd.read_json(path, lines=True)
        if path.endswith(".json"):
            return pd.read_json(path)
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        if path.endswith(".csv"):
            return pd.read_csv(path)
        raise ValueError(f"unsupported input format: {path}")

    def _export(self, df: pd.DataFrame):
        out = self.cfg["output_path"]
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
        self._log(f"exported_features={out}")

    @staticmethod
    def _safe_str_s(series: pd.Series) -> pd.Series:
        return series.astype(str)

    @staticmethod
    def _parse_listlike_count(s: pd.Series) -> pd.Series:
        # returns int count for list-like strings or lists; 0 otherwise
        def _cnt(x):
            if isinstance(x, list): return len(x)
            if isinstance(x, str):
                t = x.strip()
                if (t.startswith("[") and t.endswith("]")) or (t.startswith("(") and t.endswith(")")):
                    try:
                        v = json.loads(t.replace("(", "[").replace(")", "]"))
                        return len(v) if isinstance(v, list) else 0
                    except Exception:
                        return 0
                return 0
            return 0
        return s.apply(_cnt)

    @staticmethod
    def _resp_fields(resp: Any) -> tuple[int, Optional[int]]:
        # returns (resp_char_len, resp_unix_time or None)
        if isinstance(resp, dict):
            txt = resp.get("text", "")
            t = resp.get("time", None)
            return (len(str(txt)) if txt is not None else 0, int(t) if isinstance(t, (int, float)) else None)
        if isinstance(resp, str):
            t = resp.strip()
            if t.startswith("{") and t.endswith("}"):
                try:
                    obj = json.loads(t)
                    return FeatureEngineeringPipeline._resp_fields(obj)
                except Exception:
                    return (len(t), None)
            return (len(t), None)
        return (0, None)

    @staticmethod
    def _ascii_ratio(s: pd.Series) -> pd.Series:
        # fraction of ASCII chars
        total = s.str.len().replace(0, np.nan)
        ascii_count = s.str.count(r"[\u0000-\u007F]")
        return (ascii_count / total).fillna(0.0)

    @staticmethod
    def _nonlatin_ratio(s: pd.Series) -> pd.Series:
        # fraction of chars outside Latin (Basic + Extended A/B)
        total = s.str.len().replace(0, np.nan)
        nonlatin = s.str.count(r"[^\u0000-\u024F]")
        return (nonlatin / total).fillna(0.0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        flags = self.cfg["enable_groups"]
        short_max = int(self.cfg["length"]["short_tokens_max"])
        long_min = int(self.cfg["length"]["long_chars_min"])

        keep_cols = [c for c in ["review_id","gmap_id","user_id","time","rating","text","pics","resp"] if c in df.columns]
        out = pd.DataFrame(index=df.index)
        for c in keep_cols:
            out[c] = df[c]

        text = self._safe_str_s(df["text"])

        if flags.get("text_form", False):
            tokens = text.str.split()
            token_len = tokens.map(len)
            char_len = text.str.len()
            out["char_len"] = char_len
            out["token_len"] = token_len
            out["length_bucket"] = np.where(token_len <= short_max, "short", np.where(char_len >= long_min, "long", "medium"))
            out["emoji_count"] = count_matches(text, EMOJI_RE)
            out["punct_runs"] = count_matches(text, PUNCT_RUN_RE)
            out["elong_runs"] = count_matches(text, ELONGATED_CHAR_RE)
            out["ellipsis_runs"] = count_matches(text, ELLIPSIS_RE)
            letters = text.str.count(r"[A-Za-z]")
            uppers = text.str.count(r"[A-Z]")
            out["upper_ratio"] = (uppers / letters.replace(0, np.nan)).fillna(0.0).clip(0, 1)
            digits = text.str.count(r"\d")
            out["digit_ratio"] = (digits / char_len.replace(0, np.nan)).fillna(0.0).clip(0, 1)
            uniq_tokens = tokens.apply(lambda xs: len(set(xs)) if isinstance(xs, list) and xs else 0)
            out["unique_token_ratio"] = (uniq_tokens / token_len.replace(0, np.nan)).fillna(0.0).clip(0, 1)

        if flags.get("ad_promo", False):
            out["has_url"] = has_match(text, URL_RE)
            out["has_domain"] = has_match(text, DOMAIN_ONLY_RE)
            out["has_short_url"] = has_match(text, URL_SHORTENER_RE)
            out["has_phone_us"] = has_match(text, PHONE_US_RE)
            out["has_phone_any"] = has_match(text, GENERIC_PHONE_RE)
            out["has_email"] = has_match(text, EMAIL_RE)
            out["has_social_handle"] = has_match(text, SOCIAL_HANDLE_RE)
            out["has_hashtag"] = has_match(text, HASHTAG_RE)
            out["has_currency"] = has_match(text, CURRENCY_RE)
            out["has_percent_off"] = has_match(text, PERCENT_OFF_RE)
            out["has_promo_code"] = has_match(text, PROMO_CODE_RE)
            out["has_cta"] = has_match(text, CALL_TO_ACTION_RE)
            out["promo_hits"] = count_matches(text, self.re_promo)
            out["contact_hits"] = count_matches(text, self.re_contact)

        if flags.get("nonvisit", False):
            out["nonvisit_hits"] = count_matches(text, self.re_nonvisit)

        if flags.get("relevancy", False):
            out["offtopic_hits"] = count_matches(text, self.re_offtopic)
            out["star_rating_mention"] = has_match(text, STAR_RATING_MENTION_RE)
            # optional: if a place category column exists in merged meta
            for col in ["category","categories","types"]:
                if col in df.columns:
                    cat_s = self._safe_str_s(df[col]).str.lower()
                    # simple token overlap heuristic: any category token appears in text
                    cat_tokens = cat_s.str.replace(r"[^a-z0-9\s]", " ", regex=True).str.split()
                    out[f"mentions_{col}"] = [
                        any(tok in t.lower() for tok in (ct or []))
                        if isinstance(t, str) else False
                        for t, ct in zip(text.tolist(), cat_tokens.tolist())
                    ]
                    break

        if flags.get("media_resp", False):
            out["with_pics"] = df["pics"].notna() if "pics" in df.columns else False
            out["pic_count"] = self._parse_listlike_count(df["pics"]) if "pics" in df.columns else 0
            if "resp" in df.columns:
                resp_len, resp_time = zip(*df["resp"].apply(self._resp_fields))
                out["with_resp"] = pd.Series([rl > 0 for rl in resp_len], index=df.index)
                out["resp_char_len"] = pd.Series(resp_len, index=df.index)
                # response delay days if we have review time + resp time
                if "time" in df.columns:
                    rv_t = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
                    rs_t = pd.Series(resp_time, index=df.index)
                    rs_t = pd.to_datetime(rs_t, unit="s", utc=True, errors="coerce")
                    delay = (rs_t - rv_t).dt.total_seconds() / 86400.0
                    out["response_delay_days"] = delay.fillna(-1.0)
            else:
                out["with_resp"] = False
                out["resp_char_len"] = 0
                out["response_delay_days"] = -1.0

        if flags.get("temporal_rating", False):
            if "time" in df.columns:
                dt = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
                now = pd.Timestamp.now(tz="UTC")
                out["review_year"] = dt.dt.year.fillna(0).astype(int)
                out["review_month"] = dt.dt.month.fillna(0).astype(int)
                out["review_dow"] = dt.dt.weekday.fillna(0).astype(int)
                out["review_hour"] = dt.dt.hour.fillna(0).astype(int)
                out["age_days"] = ((now - dt).dt.total_seconds() / 86400.0).fillna(-1.0)
            else:
                for c in ["review_year","review_month","review_dow","review_hour","age_days"]:
                    out[c] = 0
            if "rating" in df.columns:
                out["extreme_star"] = df["rating"].isin([1,5]).astype(int)

        if flags.get("aggregates", False):
            if "gmap_id" in df.columns:
                grp = df.groupby("gmap_id", sort=False)
                out["place_review_count"] = grp["gmap_id"].transform("size")
                if "rating" in df.columns:
                    out["place_rating_mean"] = grp["rating"].transform("mean")
                    out["place_rating_std"] = grp["rating"].transform("std").fillna(0.0)
                    out["rating_dev_from_place_mean"] = df["rating"] - out["place_rating_mean"]
                # duplicate text count within place (spam hint)
                if "text" in df.columns:
                    dup_ct = grp["text"].transform(lambda s: s.groupby(s).transform("size"))
                    out["text_dup_count_place"] = dup_ct.fillna(1)
            if "user_id" in df.columns:
                ugr = df.groupby("user_id", sort=False)
                out["user_review_count"] = ugr["user_id"].transform("size")
                if "rating" in df.columns:
                    out["user_rating_mean"] = ugr["rating"].transform("mean")
                    out["user_rating_std"] = ugr["rating"].transform("std").fillna(0.0)
                    out["rating_dev_from_user_mean"] = df["rating"] - out["user_rating_mean"]

        if flags.get("language", False):
            out["ascii_ratio"] = self._ascii_ratio(text)
            out["nonlatin_ratio"] = self._nonlatin_ratio(text)
            thr = float(self.cfg["unicode"]["nonlatin_ratio_flag"])
            out["lang_hint_non_en"] = (out["nonlatin_ratio"] > thr).astype(int)

        self._log(f"features_rows={len(out)} cols={len(out.columns)}")
        return out

    def run(self):
        df = self._read_input()
        self._log(f"loaded_input_rows={len(df)}")
        feats = self.transform(df)
        self._export(feats)
        return feats
