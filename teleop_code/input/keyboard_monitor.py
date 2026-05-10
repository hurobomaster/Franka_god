from __future__ import annotations

import select
import sys
import termios
import tty
from typing import Optional


class KeyboardMonitor:
    """Non-blocking keyboard monitor for terminal key events."""

    def __init__(self) -> None:
        self._enabled = sys.stdin.isatty()
        self._fd: Optional[int] = None
        self._old_term = None

    def __enter__(self) -> "KeyboardMonitor":
        if self._enabled:
            self._fd = sys.stdin.fileno()
            assert self._fd is not None
            fd = self._fd
            self._old_term = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._fd is not None and self._old_term is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)

    def read_key(self) -> Optional[str]:
        if not self._enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if readable:
            return sys.stdin.read(1)
        return None
