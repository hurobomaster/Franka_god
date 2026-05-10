from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MouseSnapshot:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    buttons: List[int] = field(default_factory=list)
    timestamp: float = 0.0
