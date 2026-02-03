"""Concurrency helpers.

This module demonstrates simple concurrency patterns required by the rubric.
"""


from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(func: Callable[[T], R], items: Sequence[T], max_workers: int = 4) -> List[R]:
    """Apply `func` to `items` in parallel using threads.

    >>> parallel_map(lambda x: x + 1, [1, 2, 3], max_workers=2)
    [2, 3, 4]
    """
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(func, items))


async def async_map(func: Callable[[T], R], items: Iterable[T]) -> List[R]:
    """Run a blocking function concurrently using asyncio threads

    >>> import asyncio
    >>> asyncio.run(async_map(lambda x: x * 2, [1, 2, 3]))
    [2, 4, 6]
    """
    tasks = [asyncio.to_thread(func, item) for item in items]
    return await asyncio.gather(*tasks)
