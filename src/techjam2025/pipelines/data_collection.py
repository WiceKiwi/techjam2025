from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import glob, json
from pathlib import Path
import pandas as pd
import numpy as np

@dataclass
class MetaCfg:
    join_on: str
    keep_cols: List[str]

@dataclass
class Caps:
    max_per_place: int
    max_per_user: int

@dataclass
class Floors:
    with_pics: float
    with_resp: float
    recent: float

@dataclass
class PopTargets:
    tail: float
    mid: float
    head: float

@dataclass
class DataCollectionConfig:
    target_n: int
    random_seed: int
    sources: List[str]
    out_dir: str
    export_basename: str
    export_formats: List[str]
    caps: Caps
    floors: Floors
    popularity_targets: PopTargets
    meta: MetaCfg

    @classmethod
    def from_dict(cls, d: dict) -> "DataCollectionConfig":
        return cls(
            target_n=d["target_n"],
            random_seed=d["random_seed"],
            sources=d["sources"],
            out_dir=d["out_dir"],
            export_basename=d["export_basename"],
            export_formats=d["export_formats"],
            caps=Caps(**d["caps"]),
            floors=Floors(**d["floors"]),
            popularity_targets=PopTargets(**d["popularity_targets"]),
            meta=MetaCfg(**d["meta"]),
        )

    def sampler_cfg(self) -> dict:
        return {
            "target_n": self.target_n,
            "random_seed": self.random_seed,
            "caps": asdict(self.caps),
            "floors": asdict(self.floors),
            "popularity_targets": asdict(self.popularity_targets),
        }


class DataCollectionPipeline:
    def __init__(self, cfg: DataCollectionConfig, logger=None):
        self.cfg = cfg
        self.logger = logger

    def _log(self, msg):
        if self.logger: self.logger.info(msg)

    def _load_sources(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        paths = []
        for pat in self.cfg.sources:
            paths.extend(glob.glob(pat))
        if not paths:
            raise FileNotFoundError("no files matched in cfg.sources")

        review_paths = [p for p in paths if Path(p).name.startswith("review-")]
        meta_paths = [p for p in paths if Path(p).name.startswith("meta-")]

        if not review_paths:
            raise FileNotFoundError("no review-* files matched in cfg.sources")

        def read_any(p):
            if p.endswith(".json.gz") or p.endswith(".jsonl.gz"): return pd.read_json(p, compression="gzip", lines=True)
            if p.endswith(".jsonl"): return pd.read_json(p, lines=True)
            if p.endswith(".parquet"): return pd.read_parquet(p)
            if p.endswith(".csv") or p.endswith(".csv.gz"): return pd.read_csv(p)
            raise ValueError(f"unsupported file type: {p}")

        dfs_r = [read_any(p) for p in review_paths]
        df_reviews = pd.concat(dfs_r, ignore_index=True)

        # --- filter to reviews with non-null/non-empty text ---
        if "text" not in df_reviews.columns:
            raise ValueError("reviews missing required column 'text'")
        before = len(df_reviews)
        s = df_reviews["text"]
        s_str = s.astype(str)
        mask_text = s.notna() & s_str.str.strip().ne("") & s_str.str.lower().ne("none")
        df_reviews = df_reviews.loc[mask_text]
        dropped = before - len(df_reviews)
        self._log(f"loaded_reviews_rows={before} after_text_filter={len(df_reviews)} dropped={dropped}")

        # meta load (unchanged)
        if meta_paths:
            dfs_m = [read_any(p) for p in meta_paths]
            df_meta = pd.concat(dfs_m, ignore_index=True)
            self._log(f"loaded_meta_rows={len(df_meta)} files={len(meta_paths)}")
        else:
            df_meta = pd.DataFrame()
            self._log("no meta-* files found; continuing without meta")

        return df_reviews, df_meta

    def _export(self, df: pd.DataFrame) -> dict:
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / self.cfg.export_basename
        fmts = set(self.cfg.export_formats)
        paths = {}
        if "parquet" in fmts:
            p = f"{base}.parquet"; df.to_parquet(p, index=False); paths["parquet"] = p
        if "csv" in fmts:
            p = f"{base}.csv"; df.to_csv(p, index=False); paths["csv"] = p
        if "json" in fmts:
            p = f"{base}.json"; df.to_json(p, orient="records", force_ascii=False); paths["json"] = p
        if "jsonl" in fmts:
            p = f"{base}.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for rec in df.to_dict(orient="records"):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            paths["jsonl"] = p
        self._log(f"exported: {', '.join(paths.values())}")
        return paths

    def run(self) -> pd.DataFrame:
        df = self._load_sources()
        sample = self.stratified_sample_reviews(df, self.cfg.sampler_cfg(), logger=self.logger)
        self._export(sample)
        return sample

    @staticmethod
    def stratified_sample_reviews(df: pd.DataFrame, cfg: dict, logger=None) -> pd.DataFrame:
        # ---- validate cfg ----
        for k in ["target_n","random_seed","caps","floors","popularity_targets"]:
            if k not in cfg: raise ValueError(f"Missing config key: {k}")
        for k in ["max_per_place","max_per_user"]:
            if k not in cfg["caps"]: raise ValueError(f"Missing config key: caps.{k}")
        for k in ["with_pics","with_resp","recent"]:
            if k not in cfg["floors"]: raise ValueError(f"Missing config key: floors.{k}")
        for k in ["tail","mid","head"]:
            if k not in cfg["popularity_targets"]: raise ValueError(f"Missing config key: popularity_targets.{k}")

        target_n = int(cfg["target_n"])
        seed = int(cfg["random_seed"])
        max_per_place = int(cfg["caps"]["max_per_place"])
        max_per_user = int(cfg["caps"]["max_per_user"])
        floor_with_pics = float(cfg["floors"]["with_pics"])
        floor_with_resp = float(cfg["floors"]["with_resp"])
        floor_recent = float(cfg["floors"]["recent"])
        pop_targets = {k: float(v) for k,v in cfg["popularity_targets"].items()}

        if not {"rating","text","time","gmap_id"}.issubset(df.columns):
            raise ValueError("Input df must contain at least: rating, text, time, gmap_id")

        # ---- base filter (no copies, no mutation) ----
        mask = df["text"].astype(str).str.strip().ne("") & df["rating"].isin([1,2,3,4,5])
        if "region" in df.columns: mask &= df["region"].astype(str).str.upper().eq("AK")
        X = df.loc[mask]
        if logger: logger.info(f"pool_after_filters={len(X)} target_n={target_n}")
        if len(X) == 0: return X

        rng = np.random.RandomState(seed)

        # ---- ephemeral bins (derived from views; never assign back) ----
        text_s = X["text"].astype(str)
        token_len = text_s.str.split().map(len).to_numpy()
        char_len = text_s.str.len().to_numpy()
        length_bucket_all = np.where(token_len <= 8, "short", np.where(char_len >= 200, "long", "medium"))

        t_utc = pd.to_datetime(X["time"], unit="s", utc=True, errors="coerce")
        y = t_utc.dt.year.fillna(0).astype(int).to_numpy()
        year_bin_all = np.where(y <= 2018, "<=2018", np.where(y <= 2021, "2019-2021", np.where(y <= 2023, "2022-2023", "2024+")))
        now = pd.Timestamp.now(tz="UTC")
        recent_all = ((now - t_utc) <= pd.Timedelta(days=730)).fillna(False).to_numpy()

        with_pics_all = None
        with_resp_all = None
        if floor_with_pics > 0:
            s = X["pics"].astype(str)
            with_pics_all = (X["pics"].notna() & s.ne("None") & s.str.len().gt(2)).to_numpy()
        if floor_with_resp > 0:
            s = X["resp"].astype(str)
            with_resp_all = (X["resp"].notna() & s.ne("None") & s.str.len().gt(2)).to_numpy()

        # ---- caps ----
        def cap_group(df_in, key, kmax):
            parts = []
            for _, g in df_in.groupby(key, sort=False):
                parts.append(g if len(g) <= kmax else g.sample(n=kmax, random_state=seed))
            return pd.concat(parts, ignore_index=False)

        X = cap_group(X, "gmap_id", max_per_place)
        if "user_id" in X.columns: X = cap_group(X, "user_id", max_per_user)
        if len(X) <= target_n:
            if logger: logger.info("pool<=target; returning all rows")
            return X

        # reindex ephemeral arrays to the capped index
        base_idx = df.index[mask]
        idx = X.index
        length_bucket = pd.Series(length_bucket_all, index=base_idx).loc[idx].to_numpy()
        year_bin = pd.Series(year_bin_all, index=base_idx).loc[idx].to_numpy()
        recent = pd.Series(recent_all, index=base_idx).loc[idx].to_numpy()
        rating = X["rating"].to_numpy()
        if with_pics_all is not None:
            with_pics = pd.Series(with_pics_all, index=base_idx).loc[idx].to_numpy()
        else:
            with_pics = None
        if with_resp_all is not None:
            with_resp = pd.Series(with_resp_all, index=base_idx).loc[idx].to_numpy()
        else:
            with_resp = None

        # popularity tertiles
        place_counts = X.groupby("gmap_id").size()
        q1, q2 = place_counts.quantile([0.33,0.66]).fillna(0)
        def tier(c):
            if c <= q1: return "tail"
            if c <= q2: return "mid"
            return "head"
        pop = X["gmap_id"].map(lambda gid: tier(place_counts.get(gid, 0))).to_numpy()

        tmp = pd.DataFrame({"rating":rating,"length_bucket":length_bucket,"year_bin":year_bin,"recent":recent,"pop":pop}, index=idx)
        if with_pics is not None: tmp["with_pics"] = with_pics
        if with_resp is not None: tmp["with_resp"] = with_resp
        PRIMARY = ["rating","length_bucket","year_bin"]

        def proportional_sample(pool_idx, n):
            if len(pool_idx) <= n: return pool_idx
            g = tmp.loc[pool_idx].groupby(PRIMARY, dropna=False).size()
            total = int(g.sum())
            quotas = (g*(n/total)).round().astype(int).to_dict()
            delta = n - sum(quotas.values())
            if delta != 0:
                keys = [k for k,_ in sorted(quotas.items(), key=lambda kv: g.get(kv[0],0), reverse=True)]
                i = 0
                while delta != 0 and keys:
                    k = keys[i%len(keys)]
                    if delta > 0: quotas[k] += 1; delta -= 1
                    elif quotas[k] > 0: quotas[k] -= 1; delta += 1
                    i += 1
            picks = []
            for key,q in quotas.items():
                if q <= 0: continue
                mask_key = (tmp["rating"]==key[0]) & (tmp["length_bucket"]==key[1]) & (tmp["year_bin"]==key[2])
                cand = tmp.index[mask_key & tmp.index.isin(pool_idx)]
                if len(cand) <= q: picks.append(cand)
                else: picks.append(pd.Index(rng.choice(cand, size=q, replace=False)))
            sel = pd.Index(np.concatenate([p.values for p in picks])) if picks else pd.Index([], dtype=pool_idx.dtype)
            if len(sel) < n:
                remaining = pool_idx.difference(sel)
                if len(remaining) > 0:
                    extra = pd.Index(rng.choice(remaining, size=min(n-len(sel), len(remaining)), replace=False))
                    sel = sel.union(extra)
            return sel

        selected = pd.Index([], dtype=idx.dtype)

        def reserve(mask_bool, need):
            nonlocal selected
            if need <= 0: return
            avail = tmp.index[mask_bool & ~tmp.index.isin(selected)]
            k = min(need, len(avail))
            if k <= 0: return
            picked = proportional_sample(avail, k)
            selected = selected.union(picked)

        if floor_with_pics > 0 and "with_pics" in tmp: reserve(tmp["with_pics"], int(round(floor_with_pics*target_n)))
        if floor_with_resp > 0 and "with_resp" in tmp: reserve(tmp["with_resp"], int(round(floor_with_resp*target_n)))
        if floor_recent > 0: reserve(tmp["recent"], int(round(floor_recent*target_n)))

        tgt = {k:int(round(pop_targets[k]*target_n)) for k in ["tail","mid","head"]}
        have = tmp.loc[selected]["pop"].value_counts().to_dict()
        for tier_name in ["tail","mid","head"]:
            need = max(0, tgt[tier_name] - have.get(tier_name,0))
            if need > 0: reserve(tmp["pop"].eq(tier_name), need)

        need_final = max(0, target_n - len(selected))
        if need_final > 0:
            remainder = idx.difference(selected)
            if len(remainder) > 0:
                selected = selected.union(proportional_sample(remainder, need_final))

        if len(selected) < target_n:
            rest = idx.difference(selected)
            if len(rest) > 0:
                extra = pd.Index(rng.choice(rest, size=min(target_n-len(selected), len(rest)), replace=False))
                selected = selected.union(extra)

        return X.loc[selected, df.columns]
    
    @staticmethod
    def merge_metadata(df_reviews: pd.DataFrame, df_meta: pd.DataFrame, on: str, keep_cols: List[str] | None, logger=None) -> pd.DataFrame:
        if df_meta.empty: 
            if logger: logger.info("merge_metadata: meta empty, returning reviews unchanged")
            return df_reviews
        if keep_cols is None or len(keep_cols) == 0:
            meta_cols = [c for c in df_meta.columns if c != on and c not in df_reviews.columns]
        else:
            meta_cols = [c for c in keep_cols if c != on and (c not in df_reviews.columns or c == on)]
        # ensure unique meta rows per key
        if df_meta.duplicated(on).any():
            df_meta = df_meta.sort_values(by=on).drop_duplicates(subset=[on], keep="first")
            if logger: logger.info("merge_metadata: deduped meta on join key")
        left = len(df_reviews)
        merged = df_reviews.merge(df_meta[[on]+meta_cols], on=on, how="left", copy=False)
        matched = merged[on].notna().sum()
        if logger: logger.info(f"merge_metadata: left={left} matched_on_meta={matched}")
        return merged