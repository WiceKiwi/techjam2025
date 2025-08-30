from __future__ import annotations
import os, json, warnings
from typing import Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
    brier_score_loss
)
from sklearn.exceptions import UndefinedMetricWarning

# quiet plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _has_both_classes(y_bin: np.ndarray) -> bool:
    if y_bin.size == 0: return False
    u = np.unique(y_bin.astype(int))
    return (u.size >= 2) and (0 in u) and (1 in u)

def _safe_pr_auc(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true_bin, y_prob)) if _has_both_classes(y_true_bin) else float("nan")

def _safe_roc_auc(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true_bin, y_prob)) if _has_both_classes(y_true_bin) else float("nan")

def _safe_brier(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true_bin.astype(int), np.clip(y_prob,0,1)))
    except Exception:
        return float("nan")


class EvaluationPipeline:
    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg
        self.logger = logger
        self._validate()

    def _log(self, msg: str):
        if self.logger: self.logger.info(msg)

    @staticmethod
    def _apply_logging_controls(cfg: Dict[str, Any]):
        lg = cfg.get("logging", {})
        if lg.get("ignore_warnings", True):
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

    def _validate(self):
        need = ["data","artifacts","schema","targets","output","plots","binarization"]
        for k in need:
            if k not in self.cfg: raise ValueError(f"missing config: {k}")
        if "test_path" not in self.cfg["data"]:
            raise ValueError("missing data.test_path")
        for k in ["preds_test","thresholds_json"]:
            if k not in self.cfg["artifacts"]:
                raise ValueError(f"missing artifacts.{k}")
        if "id_key" not in self.cfg["schema"]:
            raise ValueError("missing schema.id_key")
        if not self.cfg["targets"]:
            raise ValueError("targets is empty")

    # ---- IO helpers ----
    def _read_any(self, path: str) -> pd.DataFrame:
        if path.endswith(".jsonl"): return pd.read_json(path, lines=True)
        if path.endswith(".json"):  return pd.read_json(path)
        if path.endswith(".parquet"): return pd.read_parquet(path)
        if path.endswith(".csv"): return pd.read_csv(path)
        raise ValueError(f"unsupported format: {path}")

    def _write_json(self, obj: dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _write_jsonl(self, df: pd.DataFrame, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- plotting helpers ----
    def _plot_roc_pr(self, y_true_bin, y_prob, title, out_dir, dpi):
        try:
            if not _has_both_classes(y_true_bin):
                return None, None
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
            prec, rec, _ = precision_recall_curve(y_true_bin, y_prob)
            roc_auc = _safe_roc_auc(y_true_bin, y_prob)
            pr_auc  = _safe_pr_auc(y_true_bin, y_prob)

            # ROC
            plt.figure(figsize=(4.2,3.6), dpi=dpi)
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}" if not np.isnan(roc_auc) else "AUC=NA")
            plt.plot([0,1],[0,1],'--',linewidth=1)
            plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.title(f"ROC • {title}")
            plt.legend(loc="lower right")
            p1 = Path(out_dir) / f"roc_{title}.png"
            plt.tight_layout(); plt.savefig(p1); plt.close()

            # PR
            plt.figure(figsize=(4.2,3.6), dpi=dpi)
            plt.plot(rec, prec, label=f"AP={pr_auc:.3f}" if not np.isnan(pr_auc) else "AP=NA")
            base = float(y_true_bin.mean()) if y_true_bin.size else 0.0
            plt.hlines(base, 0, 1, linestyles="--", linewidth=1)
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title(f"PR • {title}")
            plt.legend(loc="lower left")
            p2 = Path(out_dir) / f"pr_{title}.png"
            plt.tight_layout(); plt.savefig(p2); plt.close()
            return str(p1), str(p2)
        except Exception:
            return None, None

    def _plot_hist(self, y_true_bin, y_prob, title, out_dir, dpi):
        try:
            plt.figure(figsize=(4.2,3.6), dpi=dpi)
            plt.hist(y_prob[y_true_bin==1], bins=20, alpha=0.6, label="pos")
            plt.hist(y_prob[y_true_bin==0], bins=20, alpha=0.6, label="neg")
            plt.xlabel("Predicted score"); plt.ylabel("Count")
            plt.title(f"Score hist • {title}"); plt.legend()
            p = Path(out_dir) / f"hist_{title}.png"
            plt.tight_layout(); plt.savefig(p); plt.close()
            return str(p)
        except Exception:
            return None

    def _plot_calibration(self, y_true_bin, y_prob, title, out_dir, dpi, n_bins=10):
        try:
            if y_true_bin.size == 0:
                return None
            prob = np.clip(y_prob, 0, 1)
            bins = np.linspace(0, 1, n_bins+1)
            idx = np.digitize(prob, bins) - 1
            acc = []; conf = []
            for b in range(n_bins):
                mask = idx == b
                if mask.sum() == 0: continue
                conf.append(prob[mask].mean())
                acc.append(y_true_bin[mask].mean())
            plt.figure(figsize=(4.2,3.6), dpi=dpi)
            plt.plot([0,1],[0,1],'--',linewidth=1)
            if conf:
                plt.plot(conf, acc, marker="o")
            plt.xlabel("Predicted prob"); plt.ylabel("Empirical freq")
            plt.title(f"Calibration • {title}")
            p = Path(out_dir) / f"cal_{title}.png"
            plt.tight_layout(); plt.savefig(p); plt.close()
            return str(p)
        except Exception:
            return None

    # ---- eval per split ----
    def _evaluate_split(self, split_name: str, base_path: str, preds_path: str, thresholds: dict) -> pd.DataFrame:
        id_key = self.cfg["schema"]["id_key"]
        targets = self.cfg["targets"]

        base = self._read_any(base_path)
        preds = self._read_any(preds_path)
        if id_key not in preds.columns:
            raise ValueError(f"predictions missing id_key '{id_key}': {preds_path}")

        cols_needed = [id_key] + [f"pred_{t}" for t in targets]
        missing = [c for c in cols_needed if c not in preds.columns]
        if missing:
            raise ValueError(f"missing columns in predictions: {missing}")

        merged = base[[id_key] + targets].merge(preds[cols_needed], on=id_key, how="left")
        merged = merged.dropna(subset=[f"pred_{t}" for t in targets])

        rows = []
        plots_cfg = self.cfg["plots"]
        plots_dir = Path(self.cfg["output"]["plots_dir"])
        plots_dir.mkdir(parents=True, exist_ok=True)
        dpi = int(plots_cfg.get("dpi", 120))

        for tgt in targets:
            y_true = pd.to_numeric(merged[tgt], errors="coerce").fillna(0.0).values.astype(float)
            y_prob = pd.to_numeric(merged[f"pred_{tgt}"], errors="coerce").fillna(0.0).values.astype(float)
            t = float(thresholds.get(tgt, 0.5)) if self.cfg["binarization"].get("use_tuned_thresholds", True) else 0.5

            y_true_bin = (y_true >= t).astype(int)
            y_pred_bin = (np.clip(y_prob,0,1) >= t).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1]).ravel()
            prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            acc  = accuracy_score(y_true_bin, y_pred_bin)

            roc_auc = _safe_roc_auc(y_true_bin, y_prob)
            pr_auc  = _safe_pr_auc(y_true_bin, y_prob)
            brier   = _safe_brier(y_true_bin, y_prob)

            if (np.isnan(pr_auc) or np.isnan(roc_auc)) and self.cfg.get("logging",{}).get("note_on_single_class", True):
                self._log(f"note: {split_name}/{tgt} has single-class ground truth; PR/ROC set to NaN")

            # plots
            roc_path = pr_path = hist_path = cal_path = None
            title = f"{split_name}_{tgt}"
            if plots_cfg.get("roc_pr", True):
                roc_path, pr_path = self._plot_roc_pr(y_true_bin, y_prob, title, plots_dir, dpi)
            if plots_cfg.get("hist", True):
                hist_path = self._plot_hist(y_true_bin, y_prob, title, plots_dir, dpi)
            if plots_cfg.get("calibration", False):
                cal_path = self._plot_calibration(y_true_bin, y_prob, title, plots_dir, dpi)

            rows.append({
                "split": split_name,
                "target": tgt,
                "threshold": t,
                "prevalence_true": float(y_true_bin.mean()) if y_true_bin.size else 0.0,
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                "precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc),
                "roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "brier": float(brier),
                "roc_plot": roc_path, "pr_plot": pr_path, "hist_plot": hist_path, "cal_plot": cal_path
            })

        return pd.DataFrame(rows)

    # ---- main ----
    def run(self):
        self._apply_logging_controls(self.cfg)

        art = self.cfg["artifacts"]
        out = self.cfg["output"]
        Path(out["dir"]).mkdir(parents=True, exist_ok=True)
        Path(out["plots_dir"]).mkdir(parents=True, exist_ok=True)

        # thresholds (fallback to 0.5 if file missing)
        thresholds = {}
        try:
            with open(art["thresholds_json"], "r", encoding="utf-8") as f:
                thresholds = json.load(f) or {}
        except FileNotFoundError:
            self._log(f"warn: thresholds file not found ({art['thresholds_json']}); defaulting to 0.5 for all")

        frames = []
        # test split (required)
        frames.append(self._evaluate_split("test", self.cfg["data"]["test_path"], art["preds_test"], thresholds))
        # val split (optional)
        if "val_path" in self.cfg["data"] and "preds_val" in art:
            try:
                frames.append(self._evaluate_split("val", self.cfg["data"]["val_path"], art["preds_val"], thresholds))
            except Exception as e:
                self._log(f"val evaluation skipped: {e}")

        summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # save summary: JSONL + CSV + JSON
        summary_json_path = out["summary_json"]
        summary_csv_path  = out["summary_csv"]
        self._write_jsonl(summary, str(Path(summary_json_path).with_suffix(".jsonl")))
        summary.to_csv(summary_csv_path, index=False)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        # simple Markdown report
        md = ["# Evaluation Summary\n"]
        for split in summary["split"].unique():
            md.append(f"## {split}\n")
            sub = summary[summary["split"] == split].copy()
            md.append("| target | thr | prev | precision | recall | f1 | roc_auc | pr_auc | brier |")
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for _, r in sub.iterrows():
                md.append(
                    f"| {r['target']} | {r['threshold']:.3f} | {r['prevalence_true']:.3f} | "
                    f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                    f"{(r['roc_auc'] if pd.notna(r['roc_auc']) else float('nan')):.3f} | "
                    f"{(r['pr_auc'] if pd.notna(r['pr_auc']) else float('nan')):.3f} | "
                    f"{r['brier']:.3f} |"
                )
            md.append("")
        Path(out["report_md"]).parent.mkdir(parents=True, exist_ok=True)
        Path(out["report_md"]).write_text("\n".join(md), encoding="utf-8")

        self._log(f"summary(jsonl) -> {Path(summary_json_path).with_suffix('.jsonl')}")
        self._log(f"summary(csv)   -> {summary_csv_path}")
        self._log(f"report(md)     -> {out['report_md']}")
        self._log(f"plots          -> {self.cfg['output']['plots_dir']}")
