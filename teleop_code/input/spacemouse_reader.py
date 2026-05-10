from __future__ import annotations

import threading
import time
from typing import List, Optional

from input.device_state import MouseSnapshot


class SpaceMouseReader:
    """Continuously samples SpaceMouse and exposes only the latest snapshot."""

    def __init__(self, device, poll_hz: float) -> None:
        self._device = device
        self._poll_dt = 1.0 / max(1.0, poll_hz)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="spacemouse-reader", daemon=True)
        self._snapshot = MouseSnapshot(timestamp=time.perf_counter())
        self._prev_buttons: List[int] = []
        self._button_press_counts: List[int] = []
        self._error: Optional[BaseException] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def latest(self) -> tuple[MouseSnapshot, List[int], Optional[BaseException]]:
        with self._lock:
            snapshot = MouseSnapshot(
                x=self._snapshot.x,
                y=self._snapshot.y,
                z=self._snapshot.z,
                buttons=list(self._snapshot.buttons),
                timestamp=self._snapshot.timestamp,
            )
            pressed_buttons = [i for i, count in enumerate(self._button_press_counts) if count > 0]
            self._button_press_counts = [0] * len(self._button_press_counts)
            return snapshot, pressed_buttons, self._error

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = self._device.read()
                buttons = list(state.buttons)
                with self._lock:
                    if len(self._button_press_counts) < len(buttons):
                        self._button_press_counts.extend([0] * (len(buttons) - len(self._button_press_counts)))

                    for i, value in enumerate(buttons):
                        prev = self._prev_buttons[i] if i < len(self._prev_buttons) else 0
                        if value == 1 and prev == 0:
                            self._button_press_counts[i] += 1

                    self._snapshot = MouseSnapshot(
                        x=float(state.x),
                        y=float(state.y),
                        z=float(state.z),
                        buttons=buttons,
                        timestamp=time.perf_counter(),
                    )
                    self._error = None

                self._prev_buttons = buttons
            except Exception as exc:
                with self._lock:
                    self._error = exc
                time.sleep(0.05)
            else:
                time.sleep(self._poll_dt)
