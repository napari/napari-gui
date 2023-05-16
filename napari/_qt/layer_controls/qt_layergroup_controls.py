from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls

if TYPE_CHECKING:
    import napari.layers


class QtLayerGroupControls(QtLayerControls):
    layer: 'napari.layers.layergroup.LayerGroup'

    def __init__(self, layer):
        super().__init__(layer)
