import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.measure import Measure


class VispyMeasureOverlay(ViewerOverlayMixin, VispySceneOverlay):
    def __init__(self, *, viewer, overlay, parent=None) -> None:
        super().__init__(
            node=Measure(), viewer=viewer, overlay=overlay, parent=parent
        )
        self.reset()

        self.overlay.events.start.connect(self._on_data_change)
        self.overlay.events.end.connect(self._on_data_change)
        self.viewer.camera.events.zoom.connect(self._on_data_change)
        self.viewer.scale_bar.events.unit.connect(self._on_data_change)

    def _on_data_change(self):
        displayed = list(self.viewer.dims.displayed)
        start = np.array(self.overlay.start)[displayed][::-1]
        end = np.array(self.overlay.end)[displayed][::-1]

        # always pad to 3D to make handling the visual easier
        if len(displayed) == 2:
            start = np.pad(start, (0, 1))
            end = np.pad(end, (0, 1))

        length = self.overlay.length
        unit = self.viewer.scale_bar.unit

        self.node.arrow.set_data(pos=np.stack([start, end]))
        self.node.text.pos = (start + end) / 2
        self.node.text.text = f"{length} {unit}"

    def reset(self):
        super().reset()
        self._on_data_change()
