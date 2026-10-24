from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests

from .utils import ensure_dirs

BASE_URL = "https://ev.caltech.edu/api/v1/"
VALID_SITES = {"caltech", "jpl", "office001"}


def _http_date(iso_date: str) -> str:
    dt = datetime.fromisoformat(iso_date)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return format_datetime(dt, usegmt=True)


def _request_with_retry(session: requests.Session, url: str, token: str,
                        params: dict | None = None, timeout: int = 90,
                        max_attempts: int = 12):
    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.get(url, params=params, auth=(token, ""), timeout=timeout)
            if r.status_code == 401:
                raise RuntimeError("ACN API returned 401 Unauthorized. Check ACN_API_TOKEN.")
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt >= max_attempts:
                    r.raise_for_status()
                print(f"[ACN] HTTP {r.status_code}; retry {attempt}/{max_attempts} in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= max_attempts:
                raise
            print(f"[ACN] network error: {exc}\n[ACN] retry {attempt}/{max_attempts} in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError("Retry loop terminated unexpectedly.")


def _session_key(item: dict) -> str:
    return str(item.get("_id") or item.get("sessionID") or item.get("sessionId")
               or json.dumps(item, sort_keys=True, default=str))


def _download_site(token: str, site: str, out_dir: Path,
                   start: str | None, end: str | None) -> Path:
    if site not in VALID_SITES:
        raise ValueError(f"Invalid ACN site: {site}")

    final_path = out_dir / f"{site}_sessions.json"
    partial_path = out_dir / f"{site}_sessions.jsonl.part"
    checkpoint_path = out_dir / f"{site}_checkpoint.json"

    if final_path.exists() and final_path.stat().st_size > 100:
        try:
            existing = json.loads(final_path.read_text(encoding="utf-8"))
            if isinstance(existing, list) and existing:
                print(f"[ACN] {site}: existing final file ({len(existing):,} sessions) -> skipping")
                return final_path
        except Exception:
            pass

    cond = []
    if start:
        cond.append(f'connectionTime >= "{_http_date(start)}"')
    if end:
        cond.append(f'connectionTime <= "{_http_date(end)}"')

    params = {"max_results": 25, "sort": "-connectionTime"}
    if cond:
        params["where"] = " and ".join(cond)

    url = urljoin(BASE_URL, f"sessions/{site}")
    page_number = 1
    seen: set[str] = set()

    if partial_path.exists():
        with partial_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add(_session_key(json.loads(line)))
                except json.JSONDecodeError:
                    pass
        print(f"[ACN] {site}: recovered {len(seen):,} partial sessions")

    if checkpoint_path.exists():
        try:
            cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if cp.get("next_url"):
                url = cp["next_url"]
                params = None
                page_number = int(cp.get("next_page", 1))
                print(f"[ACN] {site}: resuming from page {page_number}")
        except Exception as exc:
            print(f"[ACN] {site}: checkpoint ignored: {exc}")

    session = requests.Session()
    while url:
        response = _request_with_retry(session, url, token, params=params)
        payload = response.json()
        items = payload.get("_items", [])

        if page_number == 1 and not seen and not items:
            raise RuntimeError(
                f"ACN returned zero sessions for site={site}. Refusing an empty dataset; "
                "check site/date filters/API response."
            )

        added = 0
        with partial_path.open("a", encoding="utf-8") as fh:
            for item in items:
                key = _session_key(item)
                if key in seen:
                    continue
                fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
                seen.add(key)
                added += 1

        nxt = payload.get("_links", {}).get("next", {}).get("href")
        next_url = urljoin(BASE_URL, nxt) if nxt else None
        checkpoint_path.write_text(json.dumps({
            "site": site,
            "next_page": page_number + 1,
            "downloaded_sessions": len(seen),
            "next_url": next_url,
        }, indent=2), encoding="utf-8")
        print(f"[ACN] {site}: page={page_number:,} new={added:,} total={len(seen):,}")
        url, params = next_url, None
        page_number += 1
        time.sleep(0.10)

    rows, final_seen = [], set()
    with partial_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = _session_key(item)
            if key not in final_seen:
                rows.append(item)
                final_seen.add(key)

    if not rows:
        raise RuntimeError(f"Downloaded zero sessions for {site}; refusing to write empty final dataset.")

    final_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    partial_path.unlink(missing_ok=True)
    print(f"[ACN] {site}: COMPLETE {len(rows):,} sessions -> {final_path}")
    return final_path


def download_sessions(sites: Iterable[str], out_dir: str | Path, token: str | None = None,
                      start: str | None = None, end: str | None = None) -> list[Path]:
    token = token or os.environ.get("ACN_API_TOKEN")
    if not token:
        raise RuntimeError("ACN_API_TOKEN is required. Retrieve a token from the official ACN-Data portal.")
    sites = list(sites)
    out_dir = Path(out_dir)
    ensure_dirs(out_dir)
    written = [_download_site(token, site, out_dir, start, end) for site in sites]
    if len(written) != len(sites):
        raise RuntimeError("Not all requested ACN sites were downloaded.")
    return written
