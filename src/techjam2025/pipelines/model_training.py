from __future__ import annotations
import os, json
from typing import Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, roc_auc_score, f1_score,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
import warnings
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

warnings.filterwarnings(
    "ignore",
    message="The `cv='prefit'` option is deprecated",
    category=FutureWarning
)

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
        if self.cfg["model"]["type"] not in ("lgbm",):
            raise ValueError("only model.type=lgbm supported in this pipeline")
        # Targets split by task
        if "binary" not in self.cfg["targets"] or "regression" not in self.cfg["targets"]:
            raise ValueError("targets must have 'binary' and 'regression' lists")

    # IO
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

    # feature selection
    def _pick_features(self, df: pd.DataFrame) -> List[str]:
        feats_cfg = self.cfg["features"]
        exclude = set(feats_cfg.get("exclude", []))
        mode = feats_cfg.get("mode", "auto_numeric")
        if mode == "explicit":
            feats = [c for c in feats_cfg.get("include", []) if c in df.columns and c not in exclude]
            if not feats: raise ValueError("features.include is empty or invalid")
            return feats
        numeric_like = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        drop = set(self.cfg["targets"]["binary"]) | set(self.cfg["targets"]["regression"])
        feats = [c for c in numeric_like if c not in (drop | exclude)]
        if not feats:
            raise ValueError("auto feature selection found no usable numeric columns")
        return feats

    # threshold tuner
    @staticmethod
    def _tune_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        grid = np.unique(np.clip(y_prob, 0, 1))
        if grid.size > 512:
            idx = np.linspace(0, grid.size-1, 512, dtype=int)
            grid = grid[idx]
        coarse = np.linspace(0.05, 0.95, 19)
        cand = np.unique(np.concatenate([grid, coarse]))
        best_t, best_f1 = 0.5, -1.0
        for t in cand:
            yb = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, yb, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return float(best_t)

    # metrics
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
        # Guard for single-class splits
        if len(np.unique(y_true_bin)) < 2:
            out.update({"pr_auc": float("nan"), "roc_auc": float("nan"),
                        "brier": float(brier_score_loss(y_true_bin, np.clip(y_prob,0,1)))})
            return out
        out["pr_auc"] = float(average_precision_score(y_true_bin, y_prob))
        out["roc_auc"] = float(roc_auc_score(y_true_bin, y_prob))
        out["brier"]  = float(brier_score_loss(y_true_bin, np.clip(y_prob,0,1)))
        return out

    # helpers
    def _prep_xy(self, df: pd.DataFrame, feats: List[str], tgt: str, drop_na=True):
        y = pd.to_numeric(df[tgt], errors="coerce")
        X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        if drop_na:
            m = y.notna()
            return X.loc[m], y.loc[m].values.astype(float)
        return X, y.values.astype(float)

    def run(self):
        dcfg = self.cfg["data"]
        train = self._read_any(dcfg["train_path"])
        val   = self._read_any(dcfg["val_path"])
        test  = self._read_any(dcfg["test_path"])

        self._log(f"loaded train={len(train)} val={len(val)} test={len(test)}")

        id_key = self.cfg["schema"]["id_key"]
        feats = self._pick_features(train)
        self._log(f"n_features={len(feats)}")

        Path(self.cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
        metrics_all, thresholds = {}, {}

        preds_val_blocks, preds_test_blocks = [], []

        # === Binary classifiers ===
        bin_targets = self.cfg["targets"]["binary"]
        bparams = self.cfg["model"]["binary"].get("params", {})
        class_weight = self.cfg["model"]["binary"].get("class_weight", None)
        es_rounds = int(self.cfg["train"].get("early_stopping_rounds", 50))
        calib_cfg = self.cfg["model"].get("calibration", {"enabled": False})

        for tgt in bin_targets:
            self._log(f"[bin] training: {tgt}")
            # For binaries, train on binarized silver label at 0.5 (or use your fixed prevalence cut)
            y_tr_cont = pd.to_numeric(train[tgt], errors="coerce").fillna(0.0).values
            y_va_cont = pd.to_numeric(val[tgt],   errors="coerce").fillna(0.0).values
            y_te_cont = pd.to_numeric(test[tgt],  errors="coerce").fillna(0.0).values

            y_tr = (y_tr_cont >= 0.5).astype(int)   # training labels
            X_tr = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_va = val[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_te = test[feats].replace([np.inf, -np.inf], np.nan).fillna(0)

            clf = LGBMClassifier(
                objective="binary",
                class_weight=class_weight,
                verbose=-1,
                **{k:v for k,v in bparams.items()}
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, (y_va_cont >= 0.5).astype(int))],
                callbacks=[early_stopping(stopping_rounds=es_rounds, verbose=False)],
            )

            # Optional probability calibration on val
            prob_va = clf.predict_proba(X_va)[:,1]
            if calib_cfg.get("enabled", False):
                method = calib_cfg.get("method", "isotonic")
                cal = CalibratedClassifierCV(clf, method=method, cv="prefit")
                cal.fit(X_va, (y_va_cont >= 0.5).astype(int))
                clf = cal  # replace with calibrated version
                prob_va = clf.predict_proba(X_va)[:,1]

            # tune threshold on val vs continuous target (>=t as positive)
            th_cfg = self.cfg["thresholding"]
            if th_cfg["strategy"] == "fixed":
                t = float(th_cfg["fixed"].get(tgt, 0.5))
            else:
                y_va_bin = (y_va_cont >= 0.5).astype(int)
                t = self._tune_threshold_f1(y_va_bin, prob_va)
            thresholds[tgt] = t

            # metrics
            prob_te = clf.predict_proba(X_te)[:,1]
            y_va_bin = (y_va_cont >= t).astype(int)
            y_te_bin = (y_te_cont >= t).astype(int)
            m_val = self._clf_metrics(y_va_bin, prob_va)
            m_te  = self._clf_metrics(y_te_bin, prob_te)
            metrics_all[tgt] = {"classification": {"val": m_val, "test": m_te}, "threshold": t}

            # save model
            model_path = Path(self.cfg["output"]["dir"]) / f"{tgt}_lgbm.bin.pkl"
            joblib.dump({"model": clf, "features": feats, "target": tgt, "threshold": t}, model_path)
            self._log(f"saved: {model_path}")

            preds_val_blocks.append(pd.DataFrame({id_key: val[id_key], f"pred_{tgt}": prob_va}))
            preds_test_blocks.append(pd.DataFrame({id_key: test[id_key], f"pred_{tgt}": prob_te}))

        # === Regression heads ===
        reg_targets = self.cfg["targets"]["regression"]
        rparams = self.cfg["model"]["regression"].get("params", {})

        for tgt in reg_targets:
            self._log(f"[reg] training: {tgt}")
            X_tr, y_tr = self._prep_xy(train, feats, tgt)
            X_va, y_va = self._prep_xy(val,   feats, tgt)
            X_te, y_te = self._prep_xy(test,  feats, tgt)

            reg = LGBMRegressor(objective="regression", verbose=-1, **{k:v for k,v in rparams.items()})
            reg.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[early_stopping(stopping_rounds=es_rounds, verbose=False)],
            )

            p_va = np.clip(reg.predict(X_va), 0, 1)
            p_te = np.clip(reg.predict(X_te), 0, 1)
            m_val_reg = self._reg_metrics(y_va, p_va)
            m_test_reg = self._reg_metrics(y_te, p_te)
            metrics_all[tgt] = {"regression": {"val": m_val_reg, "test": m_test_reg}}

            model_path = Path(self.cfg["output"]["dir"]) / f"{tgt}_lgbm.pkl"
            joblib.dump({"model": reg, "features": feats, "target": tgt}, model_path)
            self._log(f"saved: {model_path}")

            preds_val_blocks.append(pd.DataFrame({id_key: val[id_key], f"pred_{tgt}": p_va}))
            preds_test_blocks.append(pd.DataFrame({id_key: test[id_key], f"pred_{tgt}": p_te}))

        # merge preds
        def merge_preds(blocks: List[pd.DataFrame]) -> pd.DataFrame:
            out = blocks[0]
            for b in blocks[1:]:
                out = out.merge(b, on=id_key, how="outer")
            return out

        val_pred_df  = merge_preds(preds_val_blocks)
        test_pred_df = merge_preds(preds_test_blocks)

        # write
        Path(self.cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
        with open(self.cfg["output"]["metrics_json"], "w", encoding="utf-8") as f:
            json.dump(metrics_all, f, ensure_ascii=False, indent=2)
        with open(self.cfg["output"]["thresholds_json"], "w", encoding="utf-8") as f:
            json.dump(thresholds, f, ensure_ascii=False, indent=2)

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
