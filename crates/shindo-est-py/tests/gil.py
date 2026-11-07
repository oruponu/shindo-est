# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Checks whether a call releases the GIL, without timing it."""

from collections.abc import Callable
import sys
import threading

# Seconds; long enough that the interpreter never makes the calling thread
# hand the GIL to another thread during a test.
_SWITCH_INTERVAL = 100.0

# The calls to try, as the OS may not run the other thread in time.
_ATTEMPTS = 3


def is_enabled() -> bool:
    """Returns whether this interpreter runs with the GIL.

    Without it, another thread runs whenever it likes, so
    `is_released_during` returns True for any call.
    """
    # The public sys API has no other way to tell at run time.
    return sys._is_gil_enabled()  # pylint: disable=protected-access


def is_released_during(function: Callable[[], object]) -> bool:
    """Returns whether another Python thread runs while `function` runs.

    Another thread is made to wait for the GIL, and then `function` is
    called. The switch interval is raised meanwhile, so the other thread
    runs before `function` returns only if `function` releases the GIL.
    Even then the OS may not run it in time, so `function` is called up to
    3 times. Requires the GIL; see `is_enabled`.

    Args:
        function: The call to check.

    Returns:
        True if the other thread ran during a call, which proves that
        `function` releases the GIL.
    """
    for _ in range(_ATTEMPTS):
        if _runs_another_thread(function):
            return True
    return False


def _runs_another_thread(function: Callable[[], object]) -> bool:
    """Returns whether another thread ran during one call of `function`."""
    go = threading.Event()
    ran = threading.Event()

    def run() -> None:
        go.wait()
        ran.set()

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(_SWITCH_INTERVAL)
    try:
        thread = threading.Thread(target=run)
        thread.start()
        # The thread wakes and waits for the GIL that this thread holds.
        go.set()
        function()
        released = ran.is_set()
        thread.join()
        return released
    finally:
        sys.setswitchinterval(switch_interval)
