import re
from typing import List, Pattern
import pandas as pd

__all__ = [
    "URL_RE","PHONE_US_RE","EMAIL_RE","EMOJI_RE","PUNCT_RUN_RE","ELONGATED_CHAR_RE","ELLIPSIS_RE",
    "DOMAIN_ONLY_RE","URL_SHORTENER_RE","GENERIC_PHONE_RE","SOCIAL_HANDLE_RE","HASHTAG_RE",
    "CURRENCY_RE","PERCENT_OFF_RE","PROMO_CODE_RE","CALL_TO_ACTION_RE","STAR_RATING_MENTION_RE",
    "compile_lexicon_regex","compile_domains_regex","has_match","count_matches",
]

# URLs like http(s)://... or www....
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")

# US/CA-style phones
PHONE_US_RE = re.compile(
    r"(?x)"
    r"(?:\+?1[\s\-.]?)?"
    r"(?:\(?\d{3}\)?[\s\-.]?)"
    r"\d{3}[\s\-.]?\d{4}"
)

# Emails
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")

# Emojis (common ranges incl. flags)
EMOJI_RE = re.compile(
    "["                  # no (?i) here
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]",
    flags=re.UNICODE,
)

# Repeated punctuation runs like !!! ??? --- ___
PUNCT_RUN_RE = re.compile(r"[!?.,;:/\\|\-_*\u2026]{2,}")

# Character elongation like coooool
ELONGATED_CHAR_RE = re.compile(r"(.)\1{2,}", re.UNICODE)

# Ellipsis "..." or …
ELLIPSIS_RE = re.compile(r"(?:\.{3,}|\u2026)")

# Bare domains without scheme (ads often do this)
DOMAIN_ONLY_RE = re.compile(
    r"(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:com|net|org|io|co|ai|me|shop|store|biz|info|app|site|uk|us|ca|sg)"
    r"(?:/[^\s]*)?\b"
)

# Shorteners
URL_SHORTENER_RE = re.compile(
    r"(?i)\b(?:bit\.ly|t\.co|goo\.gl|tinyurl\.com|lnkd\.in|rebrand\.ly|is\.gd|buff\.ly|ow\.ly|cutt\.ly)/\S+\b"
)

# Generic international phone (permissive)
GENERIC_PHONE_RE = re.compile(r"(?x)(?:\+?\d[\d\-\s().]{7,}\d)")

# Social handles / hashtags
SOCIAL_HANDLE_RE = re.compile(r"(?i)(?:^|[\s:])@[\w\.]{2,}")
HASHTAG_RE = re.compile(r"(?:^|\s)#[\w]{2,}")

# Currency / % off
CURRENCY_RE = re.compile(r"(?i)\b(?:USD|SGD|US\$|S\$|\$)\s?\d{1,4}(?:[.,]\d{3})*(?:\.\d{2})?\b")
PERCENT_OFF_RE = re.compile(r"(?i)\b\d{1,3}\s?%(\s*off|\s*discount|)\b")

# Promo/coupon code mentions
PROMO_CODE_RE = re.compile(r"(?i)\b(?:use|apply|enter)\s+(?:code|coupon)\s+[A-Z0-9\-]{4,}\b")

# Call-to-action phrases
CALL_TO_ACTION_RE = re.compile(
    r"\b(?:dm|pm|message|whatsapp|telegram|call|text)\s+(?:me|us|now)\b|"
    r"\b(?:order now|limited time|link in bio)\b",
    flags=re.IGNORECASE,
)


# Explicit star-rating mentions (e.g., "4.5/5", "5 stars")
STAR_RATING_MENTION_RE = re.compile(r"(?i)\b(?:[0-5](?:\.[0-9])?\/5|[0-5](?:\.[0-9])?\s*stars?)\b")

def compile_lexicon_regex(terms: List[str]) -> Pattern:
    r"""Case-insensitive word-boundary regex for any of the given terms.
    Spaces -> \s+, hyphens -> optional [-_\s]? to catch minor variants."""
    parts = []
    for t in terms:
        t = t.strip()
        if not t:
            continue
        p = re.escape(t)
        p = p.replace(r"\ ", r"\s+")
        p = p.replace(r"\-", r"[-_\s]?")
        parts.append(p)
    if not parts:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?i)\b(?:%s)\b" % "|".join(parts))

def compile_domains_regex(tlds: List[str]) -> Pattern:
    safe = [re.escape(t.strip(". ").lower()) for t in tlds if t]
    if not safe:
        return re.compile(r"(?!x)x")
    return re.compile(
        r"(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:%s)(?:/[^\s]*)?\b" % "|".join(safe)
    )

# Vectorized helpers for pandas Series
def has_match(s: pd.Series, pattern: Pattern) -> pd.Series:
    return s.astype(str).str.contains(pattern, regex=True, na=False)

def count_matches(s: pd.Series, pattern: Pattern) -> pd.Series:
    return s.astype(str).str.count(pattern)
