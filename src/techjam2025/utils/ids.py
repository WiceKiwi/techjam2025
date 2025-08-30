import hashlib
import pandas as pd

KEYS = ["gmap_id","user_id","time","text"]

def add_review_id(df: pd.DataFrame, id_col="review_id") -> pd.DataFrame:
    if id_col in df.columns: return df
    s = pd.Series("", index=df.index)
    for k in KEYS:
        s = s + df.get(k, "").astype(str) + "||"
    s = s.str[:-2]
    df[id_col] = s.map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    return df
