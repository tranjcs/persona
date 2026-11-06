from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R], items: Iterable[T], workers: int = 8, desc: str = ""
) -> list[R | None]:
    """Map in threads, preserving order. Failures are logged and become None."""
    items = list(items)
    results: list[R | None] = [None] * len(items)
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for done, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 - keep the batch going
                failures += 1
                print(f"[{desc}] item {i} failed: {e}", file=sys.stderr)
            if done % 25 == 0 or done == len(items):
                print(f"[{desc}] {done}/{len(items)} ({failures} failed)", file=sys.stderr)
    return results
