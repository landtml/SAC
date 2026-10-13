"""On-disk cache for downloads and parsed bar frames.

The cache root is ``$SOFTEDGE_DATA`` or ``./data``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

USER_AGENT = "softedge/0.1 (+https://github.com/landtml/SAC)"
RETRY_STATUS = {429, 500, 502, 503, 504}


def cache_dir(*parts: str) -> Path:
    # SACTRADER_DATA is the name used before the project was renamed.
    root = Path(os.environ.get("SOFTEDGE_DATA") or os.environ.get("SACTRADER_DATA") or "data")
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def http_get(url: str, *, params: dict | None = None, retries: int = 4, timeout: float = 60.0,
             backoff: float = 2.0) -> requests.Response:
    """GET with retries on 429/5xx and connection errors. 404 raises ``FileNotFoundError``."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            last = exc
        else:
            if resp.status_code == 404:
                raise FileNotFoundError(url)
            if resp.status_code not in RETRY_STATUS:
                resp.raise_for_status()
                return resp
            last = requests.HTTPError(f"{resp.status_code} for {url}", response=resp)
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                time.sleep(float(retry_after))
                continue
        if attempt < retries:
            time.sleep(backoff * 2**attempt)
    raise RuntimeError(f"GET failed after {retries + 1} tries: {url}") from last


def fetch_bytes(url: str, dest: Path, **kwargs) -> bytes:
    """Return the body of ``url``, reading ``dest`` instead when it already exists."""
    if dest.exists():
        return dest.read_bytes()
    body = http_get(url, **kwargs).content
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return body


def save_frame(df: pd.DataFrame, path: Path) -> None:
    """Write a bar frame to parquet, keeping ``df.attrs``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.attrs = {}
    meta = json.dumps(df.attrs, default=str)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(out)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"softedge": meta.encode()})
    pq.write_table(table, path)


def load_frame(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    df = table.to_pandas()
    meta = table.schema.metadata or {}
    # Frames cached before the project was renamed keep their attrs under the old key.
    raw = meta.get(b"softedge") or meta.get(b"sactrader")
    df.attrs = json.loads(raw) if raw else {}
    return df
