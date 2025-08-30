# streamlit_demo.py
# Simple interactive demo for TechJam 2025 review policy classifier
# - Loads 6 trained models + thresholds from artifacts/models/
# - Loads a dataset (default: datasets/splits/test.jsonl or user upload)
# - Lets you browse reviews, see per-head scores, and the final policy decision
#
# Run:
#   streamlit run streamlit_demo.py
#
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BIN_TARGETS = ["ads_promo", "spam_low_quality", "irrelevant", "rant_no_visit"]
REG_TARGETS = ["relevancy_score", "visit_likelihood"]
ALL_TARGETS = BIN_TARGETS + REG_TARGETS

DEFAULT_MODELS_DIR = Path("artifacts/models")
DEFAULT_THRESHOLDS = DEFAULT_MODELS_DIR / "thresholds.json"
DEFAULT_TEST_PATH = Path("datasets/splits/test.jsonl")


# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dataset(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    if p.suffix == ".json":
        return pd.read_json(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported dataset format: {p.suffix}")


@st.cache_resource(show_spinner=False)
def load_model_bundle(models_dir: str | Path, target: str):
    """
    Each saved file is a joblib dump of {"model": estimator, "features": list, "target": str}
    Binary heads were saved as *_lgbm.bin.pkl ; Regression as *_lgbm.pkl
    We'll glob for either form.
    """
    models_dir = Path(models_dir)
    candidates = list(models_dir.glob(f"{target}_lgbm*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"Model file for target '{target}' not found under {models_dir}")
    # Pick the first match, but sort to make it deterministic
    model_path = sorted(candidates)[0]
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError(f"Bad model bundle at {model_path}")
    return bundle, model_path


@st.cache_resource(show_spinner=False)
def load_all_models(models_dir: str | Path):
    bundles = {}
    paths = {}
    for t in ALL_TARGETS:
        b, p = load_model_bundle(models_dir, t)
        bundles[t] = b
        paths[t] = p
    return bundles, paths


@st.cache_data(show_spinner=False)
def load_thresholds(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        # fallback to 0.5s if no thresholds file
        return {t: 0.5 for t in ALL_TARGETS}
    with open(p, "r", encoding="utf-8") as f:
        th = json.load(f)
    # fill any missing with 0.5
    for t in ALL_TARGETS:
        th.setdefault(t, 0.5)
    return th


def make_features(row: pd.Series, feat_names: list[str]) -> pd.DataFrame:
    """
    Return a *DataFrame* with the exact columns the model was trained on.
    Keeping the DataFrame (instead of .values) preserves feature names and
    avoids sklearn's 'X does not have valid feature names' warning.
    """
    X = pd.DataFrame([row]).reindex(columns=feat_names)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # ensure string column names (sometimes training artifacts store str names)
    X.columns = [str(c) for c in X.columns]
    return X


def predict_target(bundle: dict, row: pd.Series) -> float:
    model = bundle["model"]
    feats = bundle["features"]
    X = make_features(row, feats)  # keep as DataFrame with column names

    # Binary heads use calibrated LGBM with predict_proba when available
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X)[:, 1]
            return float(np.clip(prob[0], 0.0, 1.0))
        except Exception:
            pass

    # Fallback to decision_function or predict then map to [0,1]
    if hasattr(model, "decision_function"):
        v = model.decision_function(X)
        prob = 1.0 / (1.0 + np.exp(-np.clip(v, -10, 10)))
        return float(np.clip(prob[0], 0.0, 1.0))

    v = model.predict(X)
    return float(np.clip(v[0], 0.0, 1.0))


def compute_policy(scores: dict[str, float], thresholds: dict[str, float]) -> dict:
    """Return discrete flags + overall decision from continuous scores."""
    flags = {}
    # binary heads
    for t in BIN_TARGETS:
        flags[t] = int(scores.get(t, 0.0) >= float(thresholds.get(t, 0.5)))
    # relevancy logic
    irrelevant_flag = flags["irrelevant"] == 1
    relevant_by_score = scores.get("relevancy_score", 0.0) >= float(thresholds.get("relevancy_score", 0.5))
    relevancy = "irrelevant" if irrelevant_flag else ("relevant" if relevant_by_score else "irrelevant")
    # genuine
    visit_ok = scores.get("visit_likelihood", 0.0) >= float(thresholds.get("visit_likelihood", 0.5))
    genuine = int(relevancy == "relevant" and flags["ads_promo"] == 0 and flags["spam_low_quality"] == 0 and flags["rant_no_visit"] == 0 and visit_ok)
    # hold policy
    policy_hold = int(flags["spam_low_quality"] == 1 or flags["ads_promo"] == 1 or flags["irrelevant"] == 1 or flags["rant_no_visit"] == 1)
    return {
        "relevancy": relevancy,
        "genuine": genuine,
        "policy_hold": policy_hold,
        **{f"bin_{k}": v for k, v in flags.items()},
    }


def pick_id_key(df: pd.DataFrame) -> str | None:
    for c in ["review_id", "id", "row_id"]:
        if c in df.columns:
            return c
    return None


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Review Policy Demo", layout="wide")
st.title("Review Policy Demo")

# Sidebar: config
with st.sidebar:
    st.header("Configuration")
    models_dir = st.text_input("Models directory", value=str(DEFAULT_MODELS_DIR))
    thresholds_path = st.text_input("Thresholds JSON", value=str(DEFAULT_THRESHOLDS))
    st.caption("Expected files: 6 pickled model bundles + thresholds.json")

    st.divider()
    st.markdown("**Dataset**")
    use_default = st.toggle("Use default test set", value=True)
    if use_default:
        dataset_path = st.text_input("Dataset path", value=str(DEFAULT_TEST_PATH))
        uploaded = None
    else:
        uploaded = st.file_uploader("Upload dataset (.jsonl/.csv/.parquet/.json)", type=["jsonl","csv","parquet","json"], accept_multiple_files=False)
        dataset_path = None

    st.divider()
    st.caption("Tip: use the search box below to locate a review quickly.")

# Load models/thresholds
try:
    bundles, model_paths = load_all_models(models_dir)
    thresholds = load_thresholds(thresholds_path)
except Exception as e:
    st.error(f"Failed to load models/thresholds: {e}")
    st.stop()

# Load dataset
try:
    if uploaded is not None:
        # save to a temp file to let pandas sniff format
        tmp = Path(st.secrets.get("TMP_DIR", ".")) / "_tmp_uploaded_dataset"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(uploaded.getbuffer())
        df = load_dataset(str(tmp))
    else:
        df = load_dataset(dataset_path)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

if df.empty:
    st.warning("Dataset is empty.")
    st.stop()

id_key = pick_id_key(df)
if id_key is None:
    st.warning("No ID column found; using row index as ID.")
    df["_row_id"] = np.arange(len(df))
    id_key = "_row_id"

# Basic browser
left, right = st.columns([1.2, 2.0], gap="large")

with left:
    st.subheader("Browse reviews")
    txt_col_guess = None
    for c in ["text", "review", "content", "body"]:
        if c in df.columns:
            txt_col_guess = c; break
    txt_col = st.selectbox("Text column", options=[txt_col_guess] + [c for c in df.columns if c != txt_col_guess] if txt_col_guess else list(df.columns), index=0 if txt_col_guess else 0)

    q = st.text_input("Search (substring)")
    sub = df
    if q:
        try:
            sub = df[df[txt_col].astype(str).str.contains(q, case=False, na=False)]
        except Exception:
            sub = df

    st.caption(f"Showing {len(sub)} / {len(df)} rows")
    st.dataframe(sub[[id_key, txt_col]].head(200), height=320, use_container_width=True)

    selected_id = st.selectbox("Pick an ID", options=sub[id_key].head(200).tolist())
    row = sub[sub[id_key] == selected_id].iloc[0]

with right:
    st.subheader("Prediction")
    # Per-head scores
    scores = {}
    for t in ALL_TARGETS:
        try:
            scores[t] = predict_target(bundles[t], row)
        except Exception as e:
            scores[t] = float("nan")
            st.warning(f"{t}: prediction failed: {e}")

    # Display scores + thresholds
    score_rows = []
    for t in BIN_TARGETS:
        score_rows.append({"target": t, "score": scores[t], "threshold": thresholds.get(t, 0.5)})
    for t in REG_TARGETS:
        score_rows.append({"target": t, "score": scores[t], "threshold": thresholds.get(t, 0.5)})
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True)

    # Policy decision
    decision = compute_policy(scores, thresholds)
    c1, c2, c3 = st.columns(3)
    c1.metric("Relevancy", decision["relevancy"])
    c2.metric("Genuine", "Yes" if decision["genuine"] == 1 else "No")
    c3.metric("Policy Hold", "HOLD" if decision["policy_hold"] == 1 else "PASS")

    # Flags
    with st.expander("Flags (binary heads)"):
        flags_df = pd.DataFrame({
            "flag": ["ads_promo", "spam_low_quality", "irrelevant", "rant_no_visit"],
            "on": [decision["bin_ads_promo"], decision["bin_spam_low_quality"], decision["bin_irrelevant"], decision["bin_rant_no_visit"]]
        })
        st.table(flags_df)

    # Review preview
    st.markdown("#### Review")
    st.write(str(row.get(txt_col, "")))

    # Ground truth (if present)
    gt_cols = [c for c in ALL_TARGETS if c in df.columns]
    if gt_cols:
        with st.expander("Ground truth (from dataset)"):
            gt = {c: float(row.get(c)) for c in gt_cols}
            st.table(pd.DataFrame([gt]))

# Optional quick dataset-level summary
st.divider()
with st.expander("Dataset-level summary (quick)"):
    # compute policy_hold prevalence using predicted scores for first 500 rows (speed)
    preview_n = min(500, len(df))
    policy_holds = 0
    rel_counts = {"relevant": 0, "irrelevant": 0}
    for _, r in df.head(preview_n).iterrows():
        sc = {}
        ok = True
        for t in ALL_TARGETS:
            try:
                sc[t] = predict_target(bundles[t], r)
            except Exception:
                ok = False; break
        if not ok:
            continue
        dec = compute_policy(sc, thresholds)
        policy_holds += dec["policy_hold"]
        rel_counts[dec["relevancy"]] += 1
    st.write(f"Previewed {preview_n} rows")
    st.write(f"Policy HOLD rate (pred): {policy_holds/preview_n:.3f}")
    st.write(f"Relevancy split (pred): {rel_counts}")
