from __future__ import annotations
import json, math
from typing import Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm  # not used heavily, but available
except Exception:
    tqdm = None


class DatasetSplitPipeline:
    def __init__(self, cfg: Dict[str,Any], logger=None):
        self.cfg = cfg
        self.logger = logger

    def _log(self, msg: str):
        if self.logger: self.logger.info(msg)

    # ---------- IO ----------
    @staticmethod
    def _read_any(path: str) -> pd.DataFrame:
        if path.endswith(".jsonl"): return pd.read_json(path, lines=True)
        if path.endswith(".json"): return pd.read_json(path)
        if path.endswith(".parquet"): return pd.read_parquet(path)
        if path.endswith(".csv"): return pd.read_csv(path)
        raise ValueError(f"unsupported input format: {path}")

    @staticmethod
    def _write_jsonl(df: pd.DataFrame, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = df.where(pd.notnull(df), None)
        with open(path, "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- helpers ----------
    @staticmethod
    def _epoch_to_dt(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        ms = s > 1e12
        dt = pd.to_datetime(s.where(~ms, s/1000), unit="s", utc=True, errors="coerce")
        return dt

    def _ensure_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        id_key = self.cfg["id_key"]
        if id_key in df.columns and df[id_key].notna().all(): return df
        # fall back to hash of a few stable fields if present
        import hashlib
        cand = [c for c in ["review_id","gmap_id","user_id","time","text"] if c in df.columns]
        if not cand:
            base = df.index.astype(str)
            ids = base.map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
        else:
            def mk(row):
                parts = [str(row.get(c,"")) for c in cand]
                return hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()
            ids = df.apply(mk, axis=1)
        df[id_key] = ids
        self._log(f"generated {id_key} using {cand if cand else ['index']}")
        return df

    def _compute_strat_thresholds(self, df):
        prev = self.cfg["stratify"]["prevalence_pct"]
        th = {}
        for c, pct in prev.items():
            col = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            p = max(0.0, min(100.0, float(pct))) / 100.0
            q = np.clip(1.0 - p, 0.0, 1.0)
            t = float(col.quantile(q))
            if t <= 0.0:
                t = np.nextafter(0.0, 1.0)  # avoid lighting everything
            th[c] = t
        return th


    def _apply_binarization(self, df, thresholds):
        for c, t in thresholds.items():
            df[f"{c}_bin"] = (pd.to_numeric(df[c], errors="coerce") > t).astype(int)

        if self.cfg["stratify"].get("compute_genuine", True) and "relevancy_score" in df.columns:
            rel = pd.to_numeric(df["relevancy_score"], errors="coerce").fillna(0.0)
            viol = df[[f"{c}_bin" for c in thresholds]].sum(axis=1)
            df["genuine_bin"] = ((rel >= 0.6) & (viol == 0)).astype(int)
        else:
            df["genuine_bin"] = 0
        return df
    
    def _enforce_sparsity_fallbacks(self, df: pd.DataFrame) -> pd.DataFrame:
        strat = self.cfg.get("stratify", {})
        min_total = strat.get("min_total", {})
        use_rule = bool(strat.get("use_rule_proxy", True))
        topk = int(strat.get("topk_on_sparse", 0))

        # map from silver label -> rule flag column (if exists)
        rule_map = {
            "ads_promo": "rule_ads_strong",
            "spam_low_quality": "rule_spam_strong",
            "irrelevant": "rule_irrelevant_strong",
            "rant_no_visit": "rule_rant_strong",
        }

        for lbl in self.cfg["label_cols"]:
            bin_col = f"{lbl}_bin"
            need = int(min_total.get(lbl, 0))
            if need <= 0:
                continue

            have = int(df[bin_col].sum())
            if have >= need:
                continue

            # 1) OR with rule proxy if available
            if use_rule:
                rcol = rule_map.get(lbl)
                if rcol in df.columns:
                    df[bin_col] = ((df[bin_col] == 1) | (df[rcol] == 1)).astype(int)
                    have = int(df[bin_col].sum())
                    if have >= need:
                        continue

            # 2) Top-K by silver score, group-aware
            if topk > 0:
                # rank groups by their max silver score for this label
                gkey = self.cfg["group_key"]
                sc = pd.to_numeric(df[lbl], errors="coerce").fillna(0.0)
                grp_score = df.assign(_s=sc).groupby(gkey)["_s"].max().sort_values(ascending=False)
                # pick groups not already providing positives until reaching need
                pos_groups = set(df.loc[df[bin_col] == 1, gkey].unique())
                add_groups = [g for g in grp_score.index if g not in pos_groups][:max(0, need - have)]
                if add_groups:
                    df.loc[df[gkey].isin(add_groups), bin_col] = 1

        return df


    def _sample_weight(self, df: pd.DataFrame) -> pd.Series:
        mode = self.cfg["sample_weight"]["mode"]
        lbls = self.cfg["label_cols"]
        P = df[lbls].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0.0, 1.0)

        if mode == "max-violation":
            w = P.max(axis=1)
        elif mode == "1-entropy":
            eps = 1e-6
            Z = P.sum(axis=1).replace(0, np.nan)
            Q = P.div(Z, axis=0).fillna(0.0) + eps
            ent = -(Q*np.log(Q)).sum(axis=1)/math.log(Q.shape[1])
            w = 1.0 - ent
        else:
            w = pd.Series(1.0, index=df.index)

        return w.clip(0.05, 1.0)

    def _group_major_label(self, g: pd.DataFrame, lbl_cols_bin: List[str], score_cols: List[str]) -> str:
        counts = g[lbl_cols_bin].sum(axis=0)
        if counts.max() == 0:
            return "genuine"
        top = counts[counts == counts.max()].index.tolist()
        if len(top) == 1:
            return top[0].replace("_bin","")
        means = g[[c.replace("_bin","") for c in top]].mean(axis=0)
        return means.idxmax()

    def _assign_group_splits(self, df: pd.DataFrame, gkey: str, label_key: str, ratios: Dict[str,float], seed: int) -> pd.Series:
        rng = np.random.RandomState(seed)
        splits = {}

        for lab, g in df.groupby(label_key):
            gids = g[gkey].unique().tolist()
            rng.shuffle(gids)
            n = len(gids)
            n_train = int(round(ratios["train"]*n))
            n_val = int(round(ratios["val"]*n))
            train = gids[:n_train]
            val = gids[n_train:n_train+n_val]
            test = gids[n_train+n_val:]
            splits.setdefault("train", []).extend(train)
            splits.setdefault("val", []).extend(val)
            splits.setdefault("test", []).extend(test)

        grp2split = {}
        for s, gids in splits.items():
            for gid in gids: grp2split[gid] = s
        return df[gkey].map(grp2split).fillna("train")
    
    def _rebalance_min_per_split(
        self,
        df: pd.DataFrame,
        lbl_cols_bin: list,              # e.g., ["ads_promo_bin","spam_low_quality_bin",...]
        min_per_split: dict,             # e.g., {"val": 5, "test": 5}
        gkey: str,
        seed: int
    ) -> pd.DataFrame:
        """
        Greedy group-level rebalancing: move whole groups from donors into target split
        until each policy bin has at least `min_per_split[split]` positives.
        """
        rng = np.random.RandomState(seed)

        def counts_for(split: str) -> dict:
            if split not in {"train","val","test"}:
                return {c: 0 for c in lbl_cols_bin}
            sub = df[df["_split"] == split]
            if sub.empty:
                return {c: 0 for c in lbl_cols_bin}
            return sub[lbl_cols_bin].sum().astype(int).to_dict()

        # iterate target splits (only val/test are guarded)
        for split_name, need_min in (min_per_split or {}).items():
            need_min = int(need_min)
            if split_name not in {"val","test"}:
                continue

            # for each label bin, ensure coverage
            for bcol in lbl_cols_bin:
                have = counts_for(split_name).get(bcol, 0)
                if have >= need_min:
                    continue
                want = need_min - have
                if want <= 0:
                    continue

                # donors in priority order: pull from train first, then the other split
                donors = ["train", "val", "test"]
                donors.remove(split_name)

                moved = 0
                for donor in donors:
                    # candidate groups in donor split that contain at least one positive of this label
                    cand_groups = (
                        df[(df["_split"] == donor) & (df[bcol] == 1)][gkey]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    rng.shuffle(cand_groups)

                    for gid in cand_groups:
                        # move whole group to the target split
                        df.loc[df[gkey] == gid, "_split"] = split_name
                        moved += 1
                        if moved >= want:
                            break
                    if moved >= want:
                        break

                # log the result after each label
                new_have = counts_for(split_name).get(bcol, 0)
                self._log(f"rebalance: split={split_name} label={bcol} moved_groups={moved} now_have={new_have}/{need_min}")

        return df

    def run(self) -> Dict[str,str]:
        cfg = self.cfg
        path = cfg["input_path"]
        id_key = cfg["id_key"]; text_col = cfg["text_col"]; gkey = cfg["group_key"]

        df = self._read_any(path)
        self._log(f"loaded_rows={len(df)}")

        # hygiene
        df = self._ensure_ids(df)
        df = df[df[text_col].astype(str).str.strip().ne("")]
        df = df.drop_duplicates(subset=[id_key], keep="first")
        self._log(f"after_hygiene_rows={len(df)}")

        # thresholds for stratification
        thresholds = self._compute_strat_thresholds(df)
        self._log(f"stratify_thresholds={thresholds}")
        # save thresholds json for reproducibility
        out_dir = Path(cfg["outputs"]["dir"]); out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir/cfg["outputs"]["thresholds_json"], "w", encoding="utf-8") as f:
            json.dump(thresholds, f, ensure_ascii=False, indent=2)

        df = self._apply_binarization(df, thresholds)
        df = self._enforce_sparsity_fallbacks(df)

        # sample weights from silver probabilities
        df["sample_weight"] = self._sample_weight(df)

        # optional time holdout by newest groups
        df["_split"] = "unsplit"
        th_cfg = cfg.get("time_holdout", {})
        if th_cfg.get("percent", 0):
            pct = float(th_cfg["percent"]); seed = int(th_cfg["seed"])
            if cfg.get("time_col") in df.columns:
                dt = self._epoch_to_dt(df[cfg["time_col"]])
            else:
                dt = pd.Series(pd.NaT, index=df.index)
            grp_time = df.assign(_dt=dt).groupby(gkey)["_dt"].max().sort_values()
            k = max(1, int(round(pct/100.0*len(grp_time))))
            newest_groups = set(grp_time.tail(k).index.tolist())
            df.loc[df[gkey].isin(newest_groups), "_split"] = "test"
            self._log(f"time_holdout groups={len(newest_groups)} -> test")

        # group-major label for stratification on remaining
        mask = df["_split"].eq("unsplit")
        lbl_cols_bin = [f"{c}_bin" for c in cfg["label_cols"]]
        score_cols = cfg["label_cols"]
        grp_labels = {}
        for gid, g in df.groupby(gkey):
            grp_labels[gid] = self._group_major_label(g, lbl_cols_bin, score_cols)
        df["_grp_label"] = df[gkey].map(grp_labels)

        # assign remaining groups into train/val/test with label-aware proportions
        ratios = cfg["split"]["ratios"]; seed = int(cfg["split"]["seed"])
        rest = df.loc[mask, [gkey, "_grp_label"]].drop_duplicates()
        assign = self._assign_group_splits(rest, gkey=gkey, label_key="_grp_label", ratios=ratios, seed=seed)
        df.loc[mask, "_split"] = df.loc[mask, gkey].map(dict(zip(rest[gkey], assign)))
        

        # soft coverage warnings
        guard = self.cfg["stratify"].get("min_per_split", {})
        lbl_cols_bin = [f"{c}_bin" for c in self.cfg["label_cols"]]
        if guard:
            df = self._rebalance_min_per_split(
                df,
                lbl_cols_bin=lbl_cols_bin,
                min_per_split=guard,
                gkey=self.cfg["group_key"],
                seed=int(self.cfg["split"]["seed"])
            )

        for split_name, need in guard.items():
            need = int(need)
            for c in lbl_cols_bin:
                have = int(df.query("_split == @split_name")[c].sum())
                if have < need:
                    self._log(f"warn: split={split_name} class={c} have={have} < need={need}")

        # write outputs
        out_paths = {}
        for name, fname in [("train","train_name"), ("val","val_name"), ("test","test_name")]:
            p = str(out_dir/cfg["outputs"][fname])
            # drop helper columns except sample_weight and *_bin (kept for quick sanity)
            keep = [c for c in df.columns if not c.startswith("_")]
            self._write_jsonl(df[df["_split"].eq(name)][keep], p)
            out_paths[name] = p

        folds_csv = str(out_dir/cfg["outputs"]["folds_csv"])
        df[[id_key, gkey, "_grp_label", "_split"]].to_csv(folds_csv, index=False)
        out_paths["folds"] = folds_csv

        self._log(f"exported={out_paths}")
        return out_paths
