import io
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from loguru import logger
from supabase import Client, create_client


def get_supabase_client(prefer_service_role: bool = True) -> Client:
    """Create and return a Supabase client using environment variables.

    Environment variables used:
    - SUPABASE_URL
    - SUPABASE_SERVICE_ROLE_KEY (preferred for server-side ops)
      Also tolerates a mis-cased "SUPABASE_sERVICE_ROLE_KEY" if present.
    - SUPABASE_ANON_KEY (fallback if service role is unavailable)

    Args:
        prefer_service_role: Prefer using the service role key when available.

    Returns:
        A configured Supabase `Client` instance.

    Raises:
        RuntimeError: If required environment variables are missing.
    """
    # Load .env if present, resolving from current working directory upward
    load_dotenv(find_dotenv(usecwd=True))

    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    service_role_key: Optional[str] = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_sERVICE_ROLE_KEY")
    )
    anon_key: Optional[str] = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url:
        raise RuntimeError(
            "SUPABASE_URL is not set. Ensure it exists in your environment or .env file."
        )

    chosen_key: Optional[str] = None
    if prefer_service_role and service_role_key:
        chosen_key = service_role_key
        logger.debug("Using Supabase service role key.")
    else:
        chosen_key = anon_key
        logger.debug("Using Supabase anon key.")

    if not chosen_key:
        raise RuntimeError(
            "No Supabase key available. Set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY."
        )

    return create_client(supabase_url, chosen_key)


def _extract_bucket_names(buckets: Any) -> set:
    """Normalize bucket list response to a set of bucket names.

    Handles both dict-based and object-based responses.
    """
    names = set()
    for b in buckets or []:
        # supabase-py v2 returns dicts with key 'name'
        if isinstance(b, dict) and "name" in b:
            names.add(b["name"]) 
        else:
            # Fallback: try attribute access
            name = getattr(b, "name", None)
            if name:
                names.add(name)
    return names


def ensure_storage_bucket(client: Client, bucket_name: str, public: bool = False) -> None:
    """Ensure a storage bucket exists; create it if missing.

    Args:
        client: Supabase Client.
        bucket_name: Target bucket ID.
        public: Whether to create the bucket as public when creating.
    """
    existing = client.storage.list_buckets()
    existing_names = _extract_bucket_names(existing)
    if bucket_name not in existing_names:
        logger.info(f"Creating missing Supabase bucket: {bucket_name} (public={public})")
        # storage3.create_bucket signature: (id, name=None, options=None)
        # Pass public flag via options dict
        client.storage.create_bucket(bucket_name, name=bucket_name, options={"public": public})


def upload_dataframe_as_jsonl(
    dataframe: pd.DataFrame,
    bucket_name: str,
    path: Optional[str] = None,
    *,
    client: Optional[Client] = None,
    public: bool = False,
    upsert: bool = True,
    create_bucket_if_missing: bool = True,
) -> Dict[str, Any]:
    """Upload a pandas DataFrame as a JSONL file to Supabase Storage.

    Args:
        dataframe: The DataFrame to serialize and upload.
        bucket_name: The destination storage bucket name.
        path: Destination path within the bucket (e.g., "datasets/train.jsonl").
              If None, a timestamped path is generated under "data/".
        client: Optional pre-initialized Supabase client. If None, one is created.
        public: If True and the bucket is public, a public URL will be included in the result.
        upsert: If True, allow overwriting an existing file at the same path.
        create_bucket_if_missing: Create the bucket if it does not exist.

    Returns:
        Dict with keys: bucket, path, public_url (if retrievable), and raw_response.
    """
    if client is None:
        client = get_supabase_client(prefer_service_role=True)

    if path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = f"data/{timestamp}.jsonl"

    if create_bucket_if_missing:
        ensure_storage_bucket(client, bucket_name, public=public)

    jsonl_text: str = dataframe.to_json(orient="records", lines=True, force_ascii=False)
    file_bytes: bytes = jsonl_text.encode("utf-8")

    logger.info(f"Uploading DataFrame to Supabase Storage: bucket={bucket_name}, path={path}")
    storage = client.storage.from_(bucket_name)
    file_options: Dict[str, Any] = {
        "content-type": "application/json; charset=utf-8",
    }
    if upsert:
        # storage3 expects header values to be strings; it converts 'upsert' -> 'x-upsert'
        file_options["upsert"] = "true"

    response = storage.upload(path, file_bytes, file_options)

    public_url: Optional[str] = None
    try:
        # Will only be valid if the bucket/file is publicly accessible via policy
        if public:
            pub = storage.get_public_url(path)
            # Handle various return shapes from supabase-py
            if isinstance(pub, str):
                public_url = pub
            elif isinstance(pub, dict):
                public_url = (
                    pub.get("public_url")
                    or pub.get("publicUrl")
                    or (pub.get("data") or {}).get("publicUrl")
                    or (pub.get("data") or {}).get("public_url")
                )
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Public URL not available: {exc}")

    return {
        "bucket": bucket_name,
        "path": path,
        "public_url": public_url,
        "raw_response": response,
    }


def download_jsonl_to_dataframe(
    bucket_name: str,
    path: str,
    *,
    client: Optional[Client] = None,
) -> pd.DataFrame:
    """Download a JSONL file from Supabase Storage and parse into a DataFrame.

    Args:
        bucket_name: Storage bucket name.
        path: File path within the bucket.
        client: Optional pre-initialized Supabase client. If None, one is created.

    Returns:
        A pandas DataFrame parsed from the JSONL file. Returns an empty
        DataFrame if the object is empty.
    """
    if client is None:
        client = get_supabase_client(prefer_service_role=True)

    logger.info(f"Downloading JSONL from Supabase Storage: bucket={bucket_name}, path={path}")
    storage = client.storage.from_(bucket_name)
    file_bytes: bytes = storage.download(path)

    if not file_bytes:
        return pd.DataFrame()

    text = file_bytes.decode("utf-8")
    # Use pandas' JSONL reader for robustness
    return pd.read_json(io.StringIO(text), lines=True)


__all__ = [
    "get_supabase_client",
    "ensure_storage_bucket",
    "upload_dataframe_as_jsonl",
    "download_jsonl_to_dataframe",
]


