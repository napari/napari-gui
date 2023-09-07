from typing import Tuple

import numpy as np

from napari.components.overlays.base import SceneOverlay


class MeasureOverlay(SceneOverlay):
    """Measure distances in world space."""

    start: Tuple[int, ...] = (0, 0)
    end: Tuple[int, ...] = (0, 0)

    @property
    def length(self):
        """Length in world pixels."""
        return np.linalg.norm(np.array(self.end) - self.start)
