"""Project decorators"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def measure_time(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that measures execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start: float = time.time()
        result: R = func(*args, **kwargs)
        elapsed: float = time.time() - start
        print(f"[TIMER] {func.__name__} took {elapsed:.2f}s")
        return result

    return wrapper


def log_calls(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that logs function calls and arguments."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"[CALL] {func.__name__} args={args} kwargs={kwargs}")
        return func(*args, **kwargs)

    return wrapper
