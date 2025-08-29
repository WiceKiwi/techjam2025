import sys
from pathlib import Path

# Ensure "src" is on sys.path so we can import the package when running this script directly
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd
from loguru import logger

from techjam2025.utils.supabase_io import (
    upload_dataframe_as_jsonl,
    download_jsonl_to_dataframe,
)


def main() -> None:
    df = pd.DataFrame(
        [
            {"id": 1, "text": "one", "score": 0.2},
            {"id": 2, "text": "two", "score": 0.3},
            {"id": 3, "text": "three", "score": 0.4, "extra": "extra"},
        ]
    )

    # Upload. If bucket "datasets" does not exist, it will be created.
    result = upload_dataframe_as_jsonl(
        dataframe=df,
        bucket_name="datasets",
        path="data/test.jsonl",           # auto-generate timestamped path
        public=False,        # set True only if your bucket/policies allow public access
        upsert=True,
    )

    bucket = result["bucket"]
    path = result["path"]
    public_url = result.get("public_url")
    logger.info(f"Uploaded to {bucket}/{path}. Public URL: {public_url}")

    # Download and verify
    df_roundtrip = download_jsonl_to_dataframe(bucket_name=bucket, path=path)

    print("Original DataFrame:\n", df)
    print("\nDownloaded DataFrame:\n", df_roundtrip)

    try:
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True),
            df_roundtrip.reset_index(drop=True),
            check_like=True,
        )
        print("\nRound-trip check: OK ✅")
    except AssertionError as exc:
        print("\nRound-trip check: MISMATCH ❌\n", exc)


if __name__ == "__main__":
    main()


