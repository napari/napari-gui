from qtpy.QtWidgets import QFrame, QStackedWidget

from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.layer_controls.qt_layergroup_controls import (
    QtLayerGroupControls,
)
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari._qt.layer_controls.qt_tracks_controls import QtTracksControls
from napari._qt.layer_controls.qt_vectors_controls import QtVectorsControls
from napari.layers import (
    Image,
    Labels,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from napari.layers.layergroup import LayerGroup
from napari.utils import config
from napari.utils.translations import trans

layer_to_controls = {
    Labels: QtLabelsControls,
    Image: QtImageControls,
    Points: QtPointsControls,
    Shapes: QtShapesControls,
    Surface: QtSurfaceControls,
    Vectors: QtVectorsControls,
    Tracks: QtTracksControls,
    LayerGroup: QtLayerGroupControls,
}

if config.async_loading:
    from napari.layers.image.experimental.octree_image import _OctreeImageBase

    # The user visible layer controls for OctreeImage layers are identical
    # to the regular image layer controls, for now.
    layer_to_controls[_OctreeImageBase] = QtImageControls


def create_qt_layer_controls(layer):
    """
    Create a qt controls widget for a layer based on its layer type.

    In case of a subclass, the type higher in the layer's method resolution
    order will be used.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer that needs its controls widget created.

    Returns
    -------
    controls : napari.layers.base.QtLayerControls
        Qt controls widget
    """
    candidates = []
    for layer_type in layer_to_controls:
        if isinstance(layer, layer_type):
            candidates.append(layer_type)

    if not candidates:
        raise TypeError(
            trans._(
                'Could not find QtControls for layer of type {type_}',
                deferred=True,
                type_=type(layer),
            )
        )

    layer_cls = layer.__class__
    # Sort the list of candidates by 'lineage'
    candidates.sort(key=lambda layer_type: layer_cls.mro().index(layer_type))
    controls = layer_to_controls[candidates[0]]
    return controls(layer)


class QtLayerControlsContainer(QStackedWidget):
    """Container widget for QtLayerControl widgets.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    empty_widget : qtpy.QtWidgets.QFrame
        Empty placeholder frame for when no layer is selected.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    widgets : dict
        Dictionary of key value pairs matching layer with its widget controls.
        widgets[layer] = controls
    """

    def __init__(self, viewer) -> None:
        super().__init__()
        self.setProperty("emphasized", True)
        self.viewer = viewer

        self.setMouseTracking(True)
        self.empty_widget = QFrame()
        self.empty_widget.setObjectName("empty_controls_widget")
        self.widgets = {}
        self.addWidget(self.empty_widget)
        self.setCurrentWidget(self.empty_widget)

        self.viewer.layers.events.inserted.connect(self._add)
        self.viewer.layers.events.removed.connect(self._remove)
        viewer.layers.selection.events.active.connect(self._display)
        viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)

    def _on_ndisplay_changed(self, event):
        """Responds to a change in the dimensionality displayed in the canvas.

        Parameters
        ----------
        event : Event
            Event with the new dimensionality value at `event.value`.
        """
        for widget in self.widgets.values():
            if widget is not self.empty_widget:
                widget.ndisplay = event.value

    def _display(self, event):
        """Change the displayed controls to be those of the target layer.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """
        layer = event.value
        if layer is None:
            self.setCurrentWidget(self.empty_widget)
        else:
            controls = self.widgets[layer]
            self.setCurrentWidget(controls)

    def _add(self, event):
        """Add the controls target layer to the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """

        def add_children(layer):
            if layer.is_group():
                for child in layer:
                    add_children(child)
            controls = create_qt_layer_controls(layer)
            controls.ndisplay = self.viewer.dims.ndisplay
            self.addWidget(controls)
            self.widgets[layer] = controls

        add_children(event.value)

    def _remove(self, event):
        """Remove the controls target layer from the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """

        def remove_children(layer):
            if layer.is_group():
                for child in layer:
                    remove_children(child)
            controls = self.widgets[layer]
            self.removeWidget(controls)
            controls.hide()
            controls.deleteLater()
            controls = None
            del self.widgets[layer]

        remove_children(event.value)
