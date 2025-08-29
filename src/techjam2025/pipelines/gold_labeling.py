from __future__ import annotations
import hashlib, json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

class GoldLabelingPipeline:
    def __init__(self, cfg: Dict[str,Any], logger=None):
        self.cfg = cfg
        self.logger = logger
        self._validate_cfg()
        self.rng = np.random.RandomState(int(cfg["random_seed"]))

    def _log(self,msg:str):
        if self.logger: self.logger.info(msg)

    def _validate_cfg(self):
        need = ["input_rules_path","output_pack_path","random_seed","sample_sizes","stratify","dedupe","id","export_fields","labels_input_path","output_merged_path","label_schema"]
        for k in need:
            if k not in self.cfg: raise ValueError(f"missing config: {k}")
        for b in ["ads","spam","irrelevant","rant_no_visit","none"]:
            if b not in self.cfg["sample_sizes"]: raise ValueError(f"missing sample_sizes.{b}")
        for k in ["by_rating","rating_values","by_length_bucket","length_values","min_recent_frac"]:
            if k not in self.cfg["stratify"]: raise ValueError(f"missing stratify.{k}")
        for k in ["drop_exact_text_within_place"]:
            if k not in self.cfg["dedupe"]: raise ValueError(f"missing dedupe.{k}")
        for k in ["id_name","id_column","id_hash_fields"]:
            if k not in self.cfg["id"]: raise ValueError(f"missing id.{k}")
        for k in ["bool_labels","score_labels","annotator_id_col"]:
            if k not in self.cfg["label_schema"]: raise ValueError(f"missing label_schema.{k}")

    def _read_any(self,path:str) -> pd.DataFrame:
        if path.endswith(".jsonl"): return pd.read_json(path,lines=True)
        if path.endswith(".json"): return pd.read_json(path)
        if path.endswith(".parquet"): return pd.read_parquet(path)
        if path.endswith(".csv"): return pd.read_csv(path)
        raise ValueError(f"unsupported format: {path}")

    def _write_jsonl(self,df:pd.DataFrame,path:str):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        with open(path,"w",encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec,ensure_ascii=False)+"\n")

    def _stable_id(self,row:pd.Series,id_fields:List[str]) -> str:
        parts = [str(row.get(c,"")) for c in id_fields]
        h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()
        return h

    def _ensure_ids(self,df:pd.DataFrame) -> pd.DataFrame:
        id_name = self.cfg["id"]["id_name"]
        id_col = self.cfg["id"]["id_column"]
        if id_col and id_col in df.columns:
            df = df.copy()
            df.rename(columns={id_col:id_name},inplace=True)
            return df
        fields = self.cfg["id"]["id_hash_fields"]
        if not set(fields).issubset(df.columns): raise ValueError("missing fields for id hash")
        ids = df.apply(lambda r: self._stable_id(r,fields),axis=1)
        df = df.copy()
        df[id_name] = ids
        return df

    def _recent_mask(self,df:pd.DataFrame) -> pd.Series:
        if "age_days" in df.columns:
            return df["age_days"] <= 730
        if "time" in df.columns:
            dt = pd.to_datetime(df["time"],unit="s",utc=True,errors="coerce")
            now = pd.Timestamp.now(tz="UTC")
            return (now - dt).dt.total_seconds() <= 730*86400
        return pd.Series(False,index=df.index)

    def _prep_pool(self,df:pd.DataFrame) -> pd.DataFrame:
        if self.cfg["dedupe"]["drop_exact_text_within_place"] and {"gmap_id","text"}.issubset(df.columns):
            before = len(df)
            df = df.sort_index().drop_duplicates(subset=["gmap_id","text"],keep="first")
            self._log(f"dedupe_gmap_text dropped={before-len(df)}")
        need_cols = ["rating","length_bucket"]
        for c in need_cols:
            if c not in df.columns: raise ValueError(f"missing required column: {c} (ensure FeatureEngineeringPipeline ran)")
        return df

    def _bucket_filter(self,df:pd.DataFrame,bucket:str) -> pd.Series:
        m = {
            "ads":"rule_ads_strong",
            "spam":"rule_spam_strong",
            "irrelevant":"rule_irrelevant_strong",
            "rant_no_visit":"rule_rant_strong",
            "none":"rule_none",
        }
        col = m[bucket]
        if col not in df.columns: raise ValueError(f"rules column missing: {col}")
        return df[col].astype(bool)

    def _proportional_grid_sample(self,pool_idx:pd.Index,tmp:pd.DataFrame,n:int) -> pd.Index:
        if len(pool_idx) <= n: return pool_idx
        g = tmp.loc[pool_idx].groupby(["rating","length_bucket"],dropna=False).size()
        total = int(g.sum())
        quotas = (g*(n/total)).round().astype(int).to_dict()
        delta = n - sum(quotas.values())
        if delta != 0:
            keys = [k for k,_ in sorted(quotas.items(),key=lambda kv:g.get(kv[0],0),reverse=True)]
            i = 0
            while delta != 0 and keys:
                k = keys[i%len(keys)]
                if delta>0: quotas[k]+=1; delta-=1
                elif quotas[k]>0: quotas[k]-=1; delta+=1
                i+=1
        picks = []
        for key,q in quotas.items():
            if q<=0: continue
            mask = (tmp["rating"]==key[0]) & (tmp["length_bucket"]==key[1])
            cand = tmp.index[mask & tmp.index.isin(pool_idx)]
            if len(cand)<=q: picks.append(cand)
            else: picks.append(pd.Index(self.rng.choice(cand,size=q,replace=False)))
        if not picks: return pd.Index([],dtype=pool_idx.dtype)
        return pd.Index(np.concatenate([p.values for p in picks]))

    def _sample_bucket(self, df, bucket, n, recent_mask):
        idx = df.index
        tmp = pd.DataFrame({"rating": df["rating"], "length_bucket": df["length_bucket"]}, index=idx)
        want_recent = int(round(self.cfg["stratify"]["min_recent_frac"] * n))

        base = idx[self._bucket_filter(df, bucket)]
        if len(base) == 0:
            if self.cfg.get("fallback_if_no_strong", True):
                score_col = {
                    "ads": "rule_ads_score",
                    "spam": "rule_spam_score",
                    "irrelevant": "rule_irrelevant_score",
                    "rant_no_visit": "rule_rant_score",
                    "none": "rule_none",
                }[bucket]
                cand = df.index if bucket != "none" else df.index[df["rule_none"].astype(bool)]
                cand_sorted = df.loc[cand].sort_values(score_col, ascending=False).index
                return cand_sorted[:n]
            return base  # no fallback

        sel = pd.Index([], dtype=idx.dtype)
        # recent first
        recent_pool = base[recent_mask.loc[base]]
        nonrecent_pool = base.difference(recent_pool)

        k = min(want_recent, len(recent_pool))
        if k > 0:
            sel = self._proportional_grid_sample(recent_pool, tmp, k)

        need = max(0, n - len(sel))
        if need > 0 and len(nonrecent_pool) > 0:
            sel = sel.union(self._proportional_grid_sample(nonrecent_pool, tmp, need))

        # relax if still short
        need = max(0, n - len(sel))
        if need > 0:
            remaining = base.difference(sel)
            if len(remaining) > 0:
                extra = pd.Index(self.rng.choice(remaining, size=min(need, len(remaining)), replace=False))
                sel = sel.union(extra)

        need = max(0, n - len(sel))
        if need > 0:
            score_col = {
                "ads": "rule_ads_score",
                "spam": "rule_spam_score",
                "irrelevant": "rule_irrelevant_score",
                "rant_no_visit": "rule_rant_score",
                "none": "rule_none",
            }[bucket]
            cand = base.difference(sel)
            cand_sorted = df.loc[cand].sort_values(score_col, ascending=False).index
            sel = sel.union(cand_sorted[:need])

        return sel


    def build_pack(self) -> pd.DataFrame:
        df = self._read_any(self.cfg["input_rules_path"])
        self._log(f"loaded_rules_rows={len(df)}")
        df = self._ensure_ids(df)
        df = self._prep_pool(df)
        recent_mask = self._recent_mask(df)
        # make tmp for stratification grid
        if self.cfg["stratify"]["by_rating"]:
            df = df[df["rating"].isin(self.cfg["stratify"]["rating_values"])]
        if self.cfg["stratify"]["by_length_bucket"]:
            df = df[df["length_bucket"].isin(self.cfg["stratify"]["length_values"])]

        selections = []
        for bucket, n in self.cfg["sample_sizes"].items():
            sel = self._sample_bucket(df,bucket,int(n),recent_mask)
            self._log(f"bucket={bucket} need={n} got={len(sel)}")
            selections.append(sel)

        chosen = pd.Index([]).union_many(selections) if hasattr(pd.Index,"union_many") else pd.Index(np.unique(np.concatenate([s.values for s in selections if len(s)>0])))

        id_name = self.cfg["id"]["id_name"]
        fields = [c for c in self.cfg["export_fields"] if c in df.columns or c==id_name]
        pack = df.loc[chosen, fields].copy()
        # keep helpful bucket hint for audit (not shown to annotators if you prefer)
        pack["proposed_bucket"] = [
            "ads" if df.loc[i,"rule_ads_strong"] else
            "spam" if df.loc[i,"rule_spam_strong"] else
            "irrelevant" if df.loc[i,"rule_irrelevant_strong"] else
            "rant_no_visit" if df.loc[i,"rule_rant_strong"] else
            "none"
            for i in pack.index
        ]
        self._write_jsonl(pack,self.cfg["output_pack_path"])
        self._log(f"exported_pack_rows={len(pack)} -> {self.cfg['output_pack_path']}")
        return pack

    def ingest_labels(self) -> pd.DataFrame:
        id_name = self.cfg["id"]["id_name"]
        pack = self._read_any(self.cfg["output_pack_path"])
        labels = self._read_any(self.cfg["labels_input_path"])
        if id_name not in labels.columns: raise ValueError(f"labels missing id column {id_name}")

        bool_cols = self.cfg["label_schema"]["bool_labels"]
        score_cols = self.cfg["label_schema"]["score_labels"]
        annot_col = self.cfg["label_schema"]["annotator_id_col"]
        for c in bool_cols+score_cols+[annot_col]:
            if c not in labels.columns: raise ValueError(f"labels missing column: {c}")

        # group by id; majority for bools, mean for scores; track n_annot
        def agg_group(g:pd.DataFrame) -> pd.Series:
            out = {}
            for c in bool_cols:
                vc = g[c].astype(int).value_counts()
                out[c] = 1 if vc.get(1,0) >= vc.get(0,0) else 0
            for c in score_cols:
                out[c] = float(pd.to_numeric(g[c],errors="coerce").mean())
            out["n_annot"] = int(g[annot_col].nunique())
            return pd.Series(out)

        merged = labels.groupby(id_name,as_index=True).apply(agg_group)
        merged = merged.reset_index()
        out = pack.merge(merged,on=id_name,how="left")
        # sanity: label coverage
        covered = out[bool_cols].notna().all(axis=1).sum()
        self._log(f"merged_labels rows={len(out)} with_all_labels={covered}")
        self._write_jsonl(out,self.cfg["output_merged_path"])
        self._log(f"exported_merged={self.cfg['output_merged_path']}")
        return out
