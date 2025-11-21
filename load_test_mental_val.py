"""
Fire-and-forget load test for /analyze using data/mental_val.csv.

- Starts N concurrent worker threads to POST each `text` row to the FastAPI server.
- Uses a valid student_id from data/student_roster.csv (or a provided --student-id).
- Defaults to http://localhost:8000/analyze and reads data/mental_val.csv locally.
"""

import argparse
import csv
import itertools
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

try:
    import requests
except ImportError as exc:  # pragma: no cover - requests is expected to be available
    raise SystemExit("The 'requests' package is required for this script.") from exc


def load_texts(csv_path: Path, limit: int | None) -> List[str]:
    texts: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("text") or "").strip()
            if text:
                texts.append(text)
            if limit and len(texts) >= limit:
                break
    return texts


def load_student_ids(roster_path: Path) -> List[str]:
    if not roster_path.exists():
        return []
    ids: List[str] = []
    with roster_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = (row.get("student_id") or "").strip()
            if sid:
                ids.append(sid)
    return ids


def choose_student_id(ids: List[str], fallback: str) -> str:
    if ids:
        return random.choice(ids)
    return fallback


def post_one(
    session: requests.Session,
    url: str,
    student_id: str,
    text: str,
    headers: Dict[str, str],
) -> tuple[int, float]:
    payload = {"student_id": student_id, "text": text, "consent": True}
    start = time.perf_counter()
    resp = session.post(url, json=payload, headers=headers, timeout=30)
    elapsed = time.perf_counter() - start
    return resp.status_code, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Blast /analyze with mental_val.csv rows.")
    parser.add_argument("--url", default="http://localhost:8000/analyze", help="Target /analyze URL.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data") / "mental_val.csv",
        help="Path to mental_val.csv.",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path("data") / "student_roster.csv",
        help="Roster CSV for valid student IDs.",
    )
    parser.add_argument("--concurrency", type=int, default=25, help="Number of concurrent workers.")
    parser.add_argument("--limit", type=int, help="Max rows to send (default: all).")
    parser.add_argument("--student-id", help="Override student ID for every request.")
    parser.add_argument("--api-key", help="Optional X-API-Key header.")
    args = parser.parse_args()

    texts = load_texts(args.csv, args.limit)
    if not texts:
        raise SystemExit(f"No text rows found in {args.csv}")

    roster_ids = load_student_ids(args.roster)
    if not roster_ids and not args.student_id:
        raise SystemExit(
            "No student IDs found and no --student-id provided. "
            "Add IDs to the roster CSV or pass --student-id."
        )

    headers: Dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    session = requests.Session()
    lock = threading.Lock()
    successes = 0
    failures = 0
    durations: List[float] = []

    print(
        f"Sending {len(texts)} requests to {args.url} "
        f"with concurrency={args.concurrency}..."
    )

    futures = []
    id_iter = itertools.cycle(roster_ids or [args.student_id])
    for text in texts:
        sid = args.student_id or choose_student_id(roster_ids, fallback="1")
        # Step forward to vary IDs if roster is supplied
        if roster_ids:
            sid = next(id_iter)
        futures.append(
            pool.submit(post_one, session, args.url, sid, text, headers.copy())
        )

    for idx, future in enumerate(as_completed(futures), start=1):
        try:
            status, elapsed = future.result()
            with lock:
                durations.append(elapsed)
            if 200 <= status < 300:
                successes += 1
            else:
                failures += 1
        except Exception as exc:  # pragma: no cover - best-effort logging
            failures += 1
            print(f"[error] request failed: {exc}")

        if idx % 25 == 0 or idx == len(futures):
            with lock:
                avg_ms = (sum(durations) / len(durations) * 1000) if durations else 0
            print(
                f"[progress] {idx}/{len(futures)} done | ok={successes} "
                f"fail={failures} | avg={avg_ms:.1f} ms"
            )

    pool.shutdown(wait=False, cancel_futures=True)
    if durations:
        avg_ms = sum(durations) / len(durations) * 1000
        durations_sorted = sorted(durations)
        p95 = durations_sorted[int(0.95 * len(durations_sorted)) - 1] * 1000
    else:
        avg_ms = 0
        p95 = 0

    print(
        f"\nCompleted {len(futures)} requests: ok={successes}, "
        f"fail={failures}, avg={avg_ms:.1f} ms, p95={p95:.1f} ms"
    )


if __name__ == "__main__":
    main()
