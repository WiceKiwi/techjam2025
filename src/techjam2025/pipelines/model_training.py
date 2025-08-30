from __future__ import annotations
import os, json
from typing import Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_fscore_support, brier_score_loss
)
import joblib

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

class ModelTrainingPipeline:
    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg
        self.logger = logger
        self._validate()

    def _log(self, msg: str):
        if self.logger: self.logger.info(msg)

    def _validate(self):
        need = ["data","schema","targets","features","thresholding","model","train","output"]
        for k in need:
            if k not in self.cfg: raise ValueError(f"missing config: {k}")
        for k in ["train_path","val_path","test_path"]:
            if k not in self.cfg["data"]: raise ValueError(f"missing data.{k}")
        for k in ["id_key","group_key"]:
            if k not in self.cfg["schema"]: raise ValueError(f"missing schema.{k}")
        if not isinstance(self.cfg["targets"], list) or not self.cfg["targets"]:
            raise ValueError("targets must be a non-empty list")
        if self.cfg["model"]["type"] != "hgb":
            raise ValueError("only model.type=hgb supported in this pipeline")

    # ---- IO helpers ----
    def _read_any(self, path: str) -> pd.DataFrame:
        if path.endswith(".jsonl"): return pd.read_json(path, lines=True)
        if path.endswith(".json"):  return pd.read_json(path)
        if path.endswith(".parquet"): return pd.read_parquet(path)
        if path.endswith(".csv"): return pd.read_csv(path)
        raise ValueError(f"unsupported format: {path}")

    def _write_jsonl(self, df: pd.DataFrame, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = df.where(pd.notnull(df), None)
        with open(path, "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- feature selection ----
    def _pick_features(self, df: pd.DataFrame) -> List[str]:
        feats_cfg = self.cfg["features"]
        exclude = set(feats_cfg.get("exclude", []))
        mode = feats_cfg.get("mode", "auto_numeric")
        if mode == "explicit":
            feats = [c for c in feats_cfg.get("include", []) if c in df.columns and c not in exclude]
            if not feats: raise ValueError("features.include is empty or invalid")
            return feats

        # auto_numeric: pick numeric/bool columns, drop targets & excluded
        numeric_like = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        drop = set(self.cfg["targets"]) | exclude
        feats = [c for c in numeric_like if c not in drop]
        if not feats:
            raise ValueError("auto feature selection found no usable numeric columns")
        return feats

    # ---- model factory ----
    def _make_regressor(self):
        p = self.cfg["model"]["params"]
        return HistGradientBoostingRegressor(
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]),
            max_bins=int(p["max_bins"]),
            l2_regularization=float(p["l2_regularization"]),
            early_stopping=p.get("early_stopping","auto"),
            random_state=int(p["random_state"])
        )

    # ---- thresholding ----
    @staticmethod
    def _tune_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        # robust grid + unique probs
        grid = np.unique(np.clip(y_prob, 0, 1))
        if grid.size > 512:
            # subsample evenly to keep it fast
            idx = np.linspace(0, grid.size-1, 512, dtype=int)
            grid = grid[idx]
        # always include a coarse sweep
        coarse = np.linspace(0.05, 0.95, 19)
        cand = np.unique(np.concatenate([grid, coarse]))
        best_t, best_f1 = 0.5, -1.0
        for t in cand:
            yb = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, yb, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return float(best_t)

    # ---- metrics ----
    @staticmethod
    def _reg_metrics(y_true, y_pred) -> Dict[str,float]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))
        }

    @staticmethod
    def _clf_metrics(y_true_bin, y_prob) -> Dict[str,float]:
        out = {}
        try:
            out["pr_auc"] = float(average_precision_score(y_true_bin, y_prob))
        except Exception:
            out["pr_auc"] = float("nan")
        try:
            out["roc_auc"] = float(roc_auc_score(y_true_bin, y_prob))
        except Exception:
            out["roc_auc"] = float("nan")
        # Brier on probs
        try:
            out["brier"] = float(brier_score_loss(y_true_bin, np.clip(y_prob,0,1)))
        except Exception:
            out["brier"] = float("nan")
        return out

    # ---- core run ----
    def run(self):
        dcfg = self.cfg["data"]
        train = self._read_any(dcfg["train_path"])
        val   = self._read_any(dcfg["val_path"])
        test  = self._read_any(dcfg["test_path"])

        self._log(f"loaded train={len(train)} val={len(val)} test={len(test)}")

        id_key = self.cfg["schema"]["id_key"]
        gkey   = self.cfg["schema"]["group_key"]
        for df in (train,val,test):
            if id_key not in df.columns: raise ValueError(f"missing id_key in split: {id_key}")
            if gkey not in df.columns:   self._log(f"warn: group_key '{gkey}' missing in a split")

        feats = self._pick_features(train)
        self._log(f"n_features={len(feats)}")

        Path(self.cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
        metrics_all = {}
        thresholds = {}

        preds_val = []
        preds_test = []

        # train/eval per target
        for tgt in self.cfg["targets"]:
            self._log(f"==> training target: {tgt}")

            def prep(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series]:
                y = pd.to_numeric(df[tgt], errors="coerce")
                X = df[feats]
                if self.cfg["train"].get("drop_na_targets", True):
                    mask = y.notna()
                    X, y = X.loc[mask], y.loc[mask]
                    ids = df.loc[mask, id_key]
                    groups = df.loc[mask, gkey] if gkey in df.columns else pd.Series([""]*mask.sum(), index=y.index)
                else:
                    ids = df[id_key]
                    groups = df[gkey] if gkey in df.columns else pd.Series([""]*len(df), index=df.index)
                # fill NaNs in features
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                return X, y.values.astype(float), ids, groups

            X_tr, y_tr, id_tr, grp_tr = prep(train)
            X_va, y_va, id_va, grp_va = prep(val)
            X_te, y_te, id_te, grp_te = prep(test)

            reg = self._make_regressor()
            reg.fit(X_tr, y_tr)

            # predictions
            p_va = np.clip(reg.predict(X_va), 0, 1)
            p_te = np.clip(reg.predict(X_te), 0, 1)

            # regression metrics
            m_val_reg = self._reg_metrics(y_va, p_va)
            m_test_reg = self._reg_metrics(y_te, p_te)

            # thresholding for binary metrics
            th_cfg = self.cfg["thresholding"]
            if th_cfg["strategy"] == "fixed":
                t = float(th_cfg["fixed"].get(tgt, 0.5))
            else:
                # auto_f1 on val
                # need a binary ground truth from silver continuous
                # use 0.5 default or tune jointly with probs; better: auto-f1
                y_va_bin = (y_va >= 0.5).astype(int)  # serve as a reference label for tuning
                t = self._tune_threshold_f1(y_va_bin, p_va)
            thresholds[tgt] = t

            y_va_bin = (y_va >= t).astype(int)
            y_te_bin = (y_te >= t).astype(int)
            m_val_clf = self._clf_metrics(y_va_bin, p_va)
            m_test_clf = self._clf_metrics(y_te_bin, p_te)

            # pack metrics
            metrics_all[tgt] = {
                "regression": {"val": m_val_reg, "test": m_test_reg},
                "classification": {"val": m_val_clf, "test": m_test_clf},
                "threshold": t
            }

            # save model
            model_path = Path(self.cfg["output"]["dir"]) / f"{tgt}_hgb.pkl"
            joblib.dump({"model": reg, "features": feats, "target": tgt}, model_path)
            self._log(f"saved: {model_path}")

            # accumulate preds
            preds_val.append(pd.DataFrame({id_key: id_va, f"pred_{tgt}": p_va}))
            preds_test.append(pd.DataFrame({id_key: id_te, f"pred_{tgt}": p_te}))

        # merge preds per split
        def merge_preds(frames: List[pd.DataFrame]) -> pd.DataFrame:
            out = frames[0]
            for f in frames[1:]:
                out = out.merge(f, on=id_key, how="outer")
            return out

        val_pred_df = merge_preds(preds_val)
        test_pred_df = merge_preds(preds_test)

        # write outputs
        Path(self.cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
        with open(self.cfg["output"]["metrics_json"], "w", encoding="utf-8") as f:
            json.dump(metrics_all, f, ensure_ascii=False, indent=2)
        with open(self.cfg["output"]["thresholds_json"], "w", encoding="utf-8") as f:
            json.dump(thresholds, f, ensure_ascii=False, indent=2)

        # join back minimal IDs for easier downstream merges
        def join_ids(base_path, pred_df, out_path):
            base = self._read_any(base_path)[[id_key]].drop_duplicates()
            merged = base.merge(pred_df, on=id_key, how="left")
            self._write_jsonl(merged, out_path)

        join_ids(self.cfg["data"]["val_path"],  val_pred_df,  self.cfg["output"]["preds_val"])
        join_ids(self.cfg["data"]["test_path"], test_pred_df, self.cfg["output"]["preds_test"])

        self._log(f"metrics -> {self.cfg['output']['metrics_json']}")
        self._log(f"thresholds -> {self.cfg['output']['thresholds_json']}")
        self._log(f"val preds -> {self.cfg['output']['preds_val']}")
        self._log(f"test preds -> {self.cfg['output']['preds_test']}")
