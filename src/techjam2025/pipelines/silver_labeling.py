from __future__ import annotations

import os
import json
import math
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import hashlib
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

class SilverLabelingPipeline:
    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg
        self.logger = logger
        self._validate_cfg()
        self._init_model()
        self._load_prompts()

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _validate_cfg(self):
        need_top = ["input_path","output_path","sample_size","random_seed","fields","model","prompt","runtime"]
        for k in need_top:
            if k not in self.cfg:
                raise ValueError(f"missing config: {k}")
        need_fields = ["id_key","text"]
        for k in need_fields:
            if k not in self.cfg["fields"]:
                raise ValueError(f"missing fields.{k}")
        need_model = ["provider","name","max_new_tokens","temperature","top_p","retries","timeout_s"]
        for k in need_model:
            if k not in self.cfg["model"]:
                raise ValueError(f"missing model.{k}")
        need_prompt = ["system_path","few_shot_path"]
        for k in need_prompt:
            if k not in self.cfg["prompt"]:
                raise ValueError(f"missing prompt.{k}")
        # fusion optional
        if self.cfg.get("fusion", {}).get("enabled", False):
            f = self.cfg["fusion"]
            need_fusion = ["rules_path","mode","priors"]
            for k in need_fusion:
                if k not in f:
                    raise ValueError(f"missing fusion.{k}")
            if f["mode"] not in ("prior_max","blend"):
                raise ValueError("fusion.mode must be 'prior_max' or 'blend'")
            if f["mode"] == "blend" and "blend_alpha" not in f:
                raise ValueError("fusion.blend_alpha missing for mode 'blend'")
            
    def _pick_device_dtype(device_cfg: str = "auto", dtype_cfg: str = "auto"):
        try:
            import torch
            if device_cfg == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = device_cfg

            if dtype_cfg == "auto":
                if device in ("cuda", "mps"):
                    dtype = torch.float16
                else:
                    dtype = torch.float32
            else:
                dtype = getattr(torch, dtype_cfg)
            return device, dtype
        except Exception:
            # fallback if torch not installed (only makes sense for inference backend)
            return "cpu", None

    def _init_model(self):
        m = self.cfg["model"]
        if m["provider"] != "huggingface":
            raise ValueError("only huggingface provider is supported")

        backend = m.get("backend", "transformers").lower()
        if backend != "transformers":
            raise RuntimeError("Set model.backend: transformers to use local CausalLM loading")

        # local transformers imports
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

        # device & dtype (like your notebook)
        if m.get("device", "auto") == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = m["device"]

        if m.get("dtype", "auto") == "auto":
            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        else:
            dtype = getattr(torch, m["dtype"])

        self.logger.info(f"{device}")

        name = m["name"]
        trust = bool(m.get("trust_remote_code", False))
        rev = m.get("revision", None)

        # Load tokenizer/model
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust, revision=rev)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust,
            revision=rev
        ).to(device)

        # Text-generation pipeline; return only the generated continuation
        self.gen_pipe = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tok,
            return_full_text=False
        )
        self._log(f"loaded local model {name} on {device} ({dtype})")

    def _build_messages(self, row: pd.Series):
        # system: your current system prompt + hard constraint
        sys_txt = (
            self.system_text.strip() + "\n"
            "Return ONLY one valid JSON object with exactly these keys: "
            '["ads_promo","irrelevant","rant_no_visit","spam_low_quality","relevancy_score","visit_likelihood"]. '
            "No code fences, no labels, no extra text."
        )
        messages = [{"role": "system", "content": sys_txt}]

        # few-shots from your silver_fewshots file
        for fs in getattr(self, "fewshots", []):
            ins = fs.get("input", {})
            outs = fs.get("output", {})
            messages.append({"role": "user", "content": json.dumps(ins, ensure_ascii=False)})
            messages.append({"role": "assistant", "content": json.dumps(outs)})

        # minimal context + the actual review; no triple quotes, no “Example Output:”
        fields = self.cfg["fields"]
        ctx_parts = []
        pn = fields.get("place_name");  pn_val = row.get(pn) if pn else None
        if pn_val: ctx_parts.append(f"Place: {pn_val}")
        cat = fields.get("category");   cat_val = row.get(cat) if cat else None
        if cat_val: ctx_parts.append(f"Category: {cat_val}")
        rt  = fields.get("rating");     rt_val  = row.get(rt)  if rt  else None
        try:
            if rt_val is not None and not isinstance(rt_val, (list, tuple, np.ndarray)):
                ctx_parts.append(f"Rating: {int(rt_val)}")
        except Exception:
            pass
        ctx = ("\n".join(ctx_parts) + "\n") if ctx_parts else ""
        txt = str(row.get(fields["text"], "")).strip()
        user_content = f"{ctx}Review:\n{txt}"
        messages.append({"role": "user", "content": user_content})
        return messages


    def _ensure_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        id_key = self.cfg["fields"]["id_key"]
        if id_key in df.columns and df[id_key].notna().all():
            return df
        cand = [c for c in ["review_id","gmap_id","user_id","time","text"] if c in df.columns]
        if not cand:
            salt = str(self.cfg.get("input_path",""))
            ids = (df.index.astype(str) + "|" + salt).map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
        else:
            def mk(row):
                parts = [str(row.get(c,"")) for c in cand]
                return hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()
            ids = df.apply(mk, axis=1)
        out = df.copy()
        out[id_key] = ids
        self._log(f"generated_ids for {id_key} using {cand if cand else ['index+salt']}")
        return out

    def _load_prompts(self):
        p = self.cfg["prompt"]
        with open(p["system_path"], "r", encoding="utf-8") as f:
            self.system_text = f.read().strip()
        self.fewshots = []
        fp = p.get("few_shot_path","")
        if fp and Path(fp).exists():
            if fp.endswith(".jsonl"):
                with open(fp,"r",encoding="utf-8") as f:
                    self.fewshots = [json.loads(line) for line in f if line.strip()]
            elif fp.endswith(".json"):
                self.fewshots = json.load(open(fp,"r",encoding="utf-8"))
            else:
                # treat as raw text block to inline
                self.fewshot_text = Path(fp).read_text(encoding="utf-8")
        else:
            self.fewshot_text = ""

    def _read_any(self, path: str) -> pd.DataFrame:
        if path.endswith(".jsonl"):
            return pd.read_json(path, lines=True)
        if path.endswith(".json"):
            return pd.read_json(path)
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        if path.endswith(".csv"):
            return pd.read_csv(path)
        raise ValueError(f"unsupported input format: {path}")

    def _write_jsonl(self, df: pd.DataFrame, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = df.where(pd.notnull(df), None)
        with open(path, "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")


    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        if not text:
            return None
        import re
        # drop code fences and role labels the model might echo
        t = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        t = re.sub(r"(?i)\b(?:assistant|user)\s*:\s*", "", t)

        start = t.find("{")
        if start == -1:
            return None

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = t[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            # minor fix-ups: trailing commas, single quotes
                            candidate2 = re.sub(r",\s*}", "}", candidate)
                            candidate2 = candidate2.replace("'", '"')
                            try:
                                return json.loads(candidate2)
                            except Exception:
                                return None
        return None

    @staticmethod
    def _clip01(x):
        try:
            v = float(x)
            if math.isnan(v):
                return None
            return max(0.0, min(1.0, v))
        except Exception:
            return None

    def _build_prompt(self, row: pd.Series) -> str:
        # few-shots block (as simple conversation turns; no "Example Output")
        examples_block = ""
        if getattr(self, "fewshots", None):
            parts = []
            for fs in self.fewshots:
                ins = fs.get("input", {})
                outs = fs.get("output", {})
                parts.append(
                    "User: " + json.dumps(ins, ensure_ascii=False) + "\n" +
                    "Assistant: " + json.dumps(outs)
                )
            if parts:
                examples_block = "\n\n" + "\n\n".join(parts) + "\n"
        elif getattr(self, "fewshot_text", ""):
            # keep raw text if you must, but remove any 'Example Output:' strings in the file
            examples_block = "\n\n" + self.fewshot_text.strip().replace("Example Output:", "") + "\n"

        def _as_text(val):
            import numpy as _np
            if val is None: return None
            if isinstance(val, (list, tuple)): return ", ".join(map(str, val[:5])) if val else None
            if isinstance(val, _np.ndarray): return ", ".join(map(str, val.flatten().tolist()[:5])) if val.size else None
            s = str(val).strip()
            return s if s else None

        fields = self.cfg["fields"]
        ctx_parts = []

        pn_key = fields.get("place_name")
        pn_val = _as_text(row.get(pn_key)) if pn_key else None
        if pn_val: ctx_parts.append(f"Place: {pn_val}")

        cat_key = fields.get("category")
        cat_val = _as_text(row.get(cat_key)) if cat_key else None
        if cat_val: ctx_parts.append(f"Category: {cat_val}")

        rt_key = fields.get("rating")
        rt_val = row.get(rt_key) if rt_key else None
        try:
            if rt_val is not None and not isinstance(rt_val, (list, tuple, np.ndarray)):
                ctx_parts.append(f"Rating: {int(rt_val)}")
        except Exception:
            pass

        ctx = "\n".join(ctx_parts)
        ctx_section = f"\nContext:\n{ctx}\n" if ctx else ""

        text_col = fields["text"]
        txt = _as_text(row.get(text_col)) or ""

        # Build the *user* message content (system text will live in system role)
        user_block = (
            f"{examples_block}"  # examples appear *before* the real task as previous turns
            f"{ctx_section}"
            "You must reply with a single valid JSON object with these keys only: "
            '["ads_promo","irrelevant","rant_no_visit","spam_low_quality","relevancy_score","visit_likelihood"]. '
            "All values must be floats in [0,1]. Do not include code fences, quotes, or any extra text.\n\n"
            f"Review:\n{txt}"
        )
        return user_block


    def _call_model(self, messages) -> str:
        m = self.cfg["model"]
        prompt_text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        do_sample = float(m["temperature"]) > 0
        gen_kwargs = {
            "max_new_tokens": int(m["max_new_tokens"]),
            "do_sample": do_sample,
            "pad_token_id": self.tok.eos_token_id,
            # "eos_token_id": self.tok.eos_token_id,  # optional
        }
        if do_sample:
            gen_kwargs["temperature"] = float(m["temperature"])
            gen_kwargs["top_p"] = float(m["top_p"])

        outs = self.gen_pipe(prompt_text, **gen_kwargs)
        return (outs[0]["generated_text"] or "").strip()

    
    def _fuse_with_rules(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally fuse LLM scores with rule-based priors.
        Safe no-op when fusion.enabled is False or when join columns are missing.
        """
        fcfg = self.cfg.get("fusion", {})
        if not fcfg.get("enabled", False):
            return scores_df

        id_key = self.cfg["fields"]["id_key"]
        # validate inputs
        if id_key not in scores_df.columns:
            self._log(f"fusion skipped: id_key '{id_key}' not in scores_df")
            return scores_df

        try:
            rules = self._read_any(fcfg["rules_path"])
        except Exception as e:
            self._log(f"fusion skipped: cannot read rules ({e})")
            return scores_df

        needed = [id_key, "rule_ads_strong", "rule_spam_strong", "rule_irrelevant_strong", "rule_rant_strong"]
        missing = [c for c in needed if c not in rules.columns]
        if missing:
            self._log(f"fusion skipped: rules missing columns {missing}")
            return scores_df

        rules = rules[needed].set_index(id_key)
        out = scores_df.set_index(id_key)

        mode = fcfg.get("mode", "prior_max")
        alpha = float(fcfg.get("blend_alpha", 0.7)) if mode == "blend" else None
        pri = fcfg.get("priors", {})

        # ensure numeric
        for col in ["ads_promo","spam_low_quality","irrelevant","rant_no_visit"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

        mapping = {
            "ads_promo": ("rule_ads_strong", float(pri.get("ads_promo", 0.90))),
            "spam_low_quality": ("rule_spam_strong", float(pri.get("spam_low_quality", 0.85))),
            "irrelevant": ("rule_irrelevant_strong", float(pri.get("irrelevant", 0.80))),
            "rant_no_visit": ("rule_rant_strong", float(pri.get("rant_no_visit", 0.80))),
        }

        for label, (flag, prior_val) in mapping.items():
            if label not in out.columns or flag not in rules.columns:
                continue
            mask = rules[flag] == 1
            if mode == "prior_max":
                out.loc[mask, label] = np.maximum(out.loc[mask, label].astype(float), prior_val)
            else:  # blend
                out.loc[mask, label] = float(alpha) * prior_val + (1.0 - float(alpha)) * out.loc[mask, label].astype(float)

        return out.reset_index()
    
    def _debug_dir(self):
        dd = self.cfg.get("runtime", {}).get("debug_dump", {})
        if not dd.get("enabled", False): return None
        root = Path(dd.get("dir", "debug/silver"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = root / stamp
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _maybe_dump(self, dbg_dir, idx, review_id, prompt, raw):
        dd = self.cfg.get("runtime", {}).get("debug_dump", {})
        if not dbg_dir: return
        if idx >= int(dd.get("max_items", 10)): return
        (dbg_dir / f"{idx:03d}_{review_id}_raw.txt").write_text(str(raw or ""), encoding="utf-8")
        if dd.get("include_prompts", True):
            (dbg_dir / f"{idx:03d}_{review_id}_prompt.txt").write_text(prompt, encoding="utf-8")

    def _repair_to_json(self, text: str) -> Optional[dict]:
        schema = {
            "ads_promo": 0.0, "irrelevant": 0.0, "rant_no_visit": 0.0,
            "spam_low_quality": 0.0, "relevancy_score": 0.0, "visit_likelihood": 0.0
        }
        try:
            repair_prompt = (
                f"{self.system_text}\n\n"
                "Return ONLY a valid JSON object with these exact keys and values in [0,1]:\n"
                f"{json.dumps(schema)}\n\nReview:\n\"\"\"{text.strip()}\"\"\""
            )
            raw = self._call_model(repair_prompt)
            return self._extract_json(raw)
        except Exception:
            return None
        
    def merge_with_input(self, base_path: str = None, silver_path: str = None, out_path: str = None, how: str = "left") -> pd.DataFrame:
        """
        Merge the silver labels onto the original input via id_key,
        using SILVER as the left table (so output ~ size of silver).
        """
        id_key = self.cfg["fields"]["id_key"]
        base_path = base_path or self.cfg["input_path"]
        silver_path = silver_path or self.cfg["output_path"]

        base = self._read_any(base_path)
        if id_key not in base.columns:
            base = self._ensure_ids(base)

        silver = self._read_any(silver_path)
        if id_key not in silver.columns:
            raise ValueError(f"silver file missing {id_key}: {silver_path}")

        # de-dup to avoid row explosion
        if silver[id_key].duplicated().any():
            self._log("duplicates in silver; keeping first per id")
            silver = silver.drop_duplicates(id_key, keep="first")
        if base[id_key].duplicated().any():
            self._log("duplicates in base; keeping first per id")
            base = base.drop_duplicates(id_key, keep="first")

        # silver-driven merge
        merged = silver.merge(base, on=id_key, how=how, suffixes=("", "_base"))

        silver_cols = [c for c in silver.columns if c != id_key]
        base_cols = [c for c in base.columns if c != id_key and c not in silver_cols]
        merged = merged[[id_key] + silver_cols + base_cols]

        if out_path is None:
            p = Path(silver_path)
            out_path = str(p.with_name(p.stem + "_merged.jsonl"))

        self._write_jsonl(merged, out_path)
        self._log(f"exported_merged={out_path} rows={len(merged)}")
        return merged


    def run(self) -> pd.DataFrame:
        df = self._read_any(self.cfg["input_path"])
        self._log(f"loaded_rows={len(df)}")

        n = int(self.cfg["sample_size"])
        if n and len(df) > n:
            df = df.sample(n, random_state=int(self.cfg["random_seed"]))
            self._log(f"sampled_rows={len(df)}")

        df = self._ensure_ids(df)
        id_key = self.cfg["fields"]["id_key"]
        text_col = self.cfg["fields"]["text"]
        if text_col not in df.columns:
            raise ValueError(f"missing text column: {text_col}")
        if id_key not in df.columns:
            self._log(f"warning: id_key '{id_key}' not found; will emit empty ids")

        dbg_dir = self._debug_dir()

        rows = []
        
        records = df.to_dict(orient="records")
        iterator = tqdm(records, total=len(records),
                        disable=not self.cfg["runtime"].get("use_tqdm", True) or tqdm is None)
        for idx, row in enumerate(iterator):
            sr = pd.Series(row)

            try:
                messages = self._build_messages(sr)
                raw_text = self._call_model(messages)
                js = self._extract_json(raw_text or "")
                if js is None and self.cfg["runtime"].get("json_guard", True):
                    txt = str(sr.get(self.cfg["fields"]["text"], "")).strip()
                    js = self._repair_to_json(txt) or None
            except Exception as e:
                js = None
                self._log(f"model_error: {e}")

            # Hard fallback to valid numbers (prevents NaN)
            if js is None:
                js = {"ads_promo": 0.0, "irrelevant": 0.0, "rant_no_visit": 0.0,
                    "spam_low_quality": 0.0, "relevancy_score": 0.0, "visit_likelihood": 0.0}

            rec = {
                self.cfg["fields"]["id_key"]: sr.get(self.cfg["fields"]["id_key"], ""),
                "ads_promo": self._clip01(js.get("ads_promo")),
                "irrelevant": self._clip01(js.get("irrelevant")),
                "rant_no_visit": self._clip01(js.get("rant_no_visit")),
                "spam_low_quality": self._clip01(js.get("spam_low_quality")),
                "relevancy_score": self._clip01(js.get("relevancy_score")),
                "visit_likelihood": self._clip01(js.get("visit_likelihood")),
            }

            # Store a canonical JSON string as raw_model (strict JSON, not code-fenced junk)
            rec["raw_model"] = json.dumps({
                "ads_promo": rec["ads_promo"],
                "irrelevant": rec["irrelevant"],
                "rant_no_visit": rec["rant_no_visit"],
                "spam_low_quality": rec["spam_low_quality"],
                "relevancy_score": rec["relevancy_score"],
                "visit_likelihood": rec["visit_likelihood"],
            }, ensure_ascii=False)

            rows.append(rec)
            self._maybe_dump(dbg_dir, idx, rec.get(self.cfg["fields"]["id_key"], "noid"),
                            json.dumps(messages, ensure_ascii=False), raw_text)


        out = pd.DataFrame(rows)
        out = out.drop(columns=["raw_model"], errors="ignore")
        if self.cfg.get("fusion", {}).get("enabled", False):
            out = self._fuse_with_rules(out)
        self._write_jsonl(out, self.cfg["output_path"])
        self._log(f"exported_silver={self.cfg['output_path']}")
        return out

