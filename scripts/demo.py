import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml

def _read_any(path):
    p = str(path)
    if not Path(p).exists():
        return None
    if p.endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    if p.endswith(".json"):
        return pd.read_json(p)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith(".csv"):
        return pd.read_csv(p)
    return None

def _safe_head(df, n=1):
    try:
        return df.head(n)
    except Exception:
        return None

def _short(s, n):
    s = str(s or "")
    return (s[:n] + "…") if len(s) > n else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    paths = cfg["paths"]
    targets = cfg.get("display", {}).get("targets", [])
    sample_chars = int(cfg.get("display", {}).get("sample_chars", 140))
    kdec = int(cfg.get("display", {}).get("sample_decisions", 1))

    print("\n=== 1) Silver label sample ===")
    silver = _read_any(paths["silver_sample"])
    if silver is not None and len(silver):
        row = silver.iloc[0].to_dict()
        text = row.get("text", "")
        scores = {t: row.get(t, None) for t in targets if t in row}
        print(f"review_id: {row.get('review_id','')}")
        print("text:", _short(text, sample_chars))
        print("scores:", {k: (None if v is None else round(float(v),3)) for k,v in scores.items()})
    else:
        print("silver sample missing")

    print("\n=== 2) Split sizes ===")
    for name in ("train","val","test"):
        df = _read_any(paths[name])
        print(f"{name}: {0 if df is None else len(df)} rows")

    print("\n=== 3) Evaluation (test) ===")
    eval_sum = _read_any(paths["eval_summary"])
    if eval_sum is not None and len(eval_sum):
        test = eval_sum[eval_sum["split"]=="test"].copy()
        keep = ["target","threshold","prevalence_true","precision","recall","f1","pr_auc","roc_auc","brier"]
        test = test[keep] if all(c in test.columns for c in keep) else test
        def _fmt(x):
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)
        for _,r in test.iterrows():
            print(
                f"{r['target']:<18} thr={_fmt(r['threshold'])} "
                f"prev={_fmt(r['prevalence_true'])} "
                f"P={_fmt(r['precision'])} R={_fmt(r['recall'])} "
                f"F1={_fmt(r['f1'])} PR-AUC={_fmt(r['pr_auc'])}"
            )
    else:
        print("evaluation summary missing")

    print("\n=== 4) Policy decisions ===")
    counts = _read_any(paths["policy_counts"])
    if counts is not None and len(counts):
        print(counts.to_string(index=False))
    else:
        print("policy counts missing")

    decisions = _read_any(paths["policy_decisions"])
    if decisions is not None and len(decisions):
        print("\nSample decision:")
        for _,r in decisions.head(kdec).iterrows():
            d = r.to_dict()
            rid = d.get("review_id","")
            act = d.get("final_action","")
            rsn = d.get("reasons",[])
            print(json.dumps({"review_id": rid, "action": act, "reasons": rsn}, ensure_ascii=False))
    else:
        print("policy decisions missing")

    print("\n(Thresholds are configurable; lower spam threshold → more REVIEW/REMOVE. End.)")

if __name__ == "__main__":
    sys.exit(main())
