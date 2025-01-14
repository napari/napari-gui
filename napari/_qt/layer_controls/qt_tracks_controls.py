from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets._tracks import (
    QtColormapComboBoxControl,
    QtColorPropertiesComboBoxControl,
    QtGraphCheckBoxControl,
    QtHeadLengthSliderControl,
    QtIdCheckBoxControl,
    QtTailDisplayCheckBoxControl,
    QtTailLengthSliderControl,
    QtTailWidthSliderControl,
)
from napari.layers.base._base_constants import Mode

if TYPE_CHECKING:
    import napari.layers


class QtTracksControls(QtLayerControls):
    """Qt view and controls for the Tracks layer.

    Parameters
    ----------
    layer : napari.layers.Tracks
        An instance of a Tracks layer.

    Attributes
    ----------
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of points layer modes (ADD, PAN_ZOOM, SELECT).
    layer : layers.Tracks
        An instance of a Tracks layer.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button for pan/zoom mode.
    qtColormapComboBoxControl.colormap_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current colormap of the layer.
    qtColormapComboBoxControl.colormap_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the colormap chooser widget.
    qtGraphCheckBoxControl.graph_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if graph of the layer should be shown.
    qtGraphCheckBoxControl.graph_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the graph chooser widget.
    qtHeadLengthSliderControl.head_length_slider : qtpy.QtWidgets.QSlider
        Slider controlling head length of the layer.
    qtHeadLengthSliderControl.head_length_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the head length chooser widget.
    qtIdCheckBoxControl.id_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if id of the layer should be shown.
    qtIdCheckBoxControl.id_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the id chooser widget.
    qtOpacityBlendingControls.blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    qtOpacityBlendingControls.blendLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    qtOpacityBlendingControls.opacityLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    qtOpacityBlendingControls.opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    qtTailDisplayCheckBoxControl.tail_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if tails of the layer should be shown.
    qtTailDisplayCheckBoxControl.tail_tail_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the tails chooser widget.
    qtTailLengthSliderControl.tail_length_slider : qtpy.QtWidgets.QSlider
        Slider controlling tail length of the layer.
    qtTailLengthSliderControl.tail_length_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the tail length chooser widget.
    qtTailWidthSliderControl.tail_width_slider : qtpy.QtWidgets.QSlider
        Slider controlling tail width of the layer.
    qtTailWidthSliderControl.tail_width_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the tail width chooser widget.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select transform mode.
    """

    layer: 'napari.layers.Tracks'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_tracks_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_tracks_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._add_widget_controls(
            QtColorPropertiesComboBoxControl(self, layer)
        )
        self._add_widget_controls(QtColormapComboBoxControl(self, layer))
        self._add_widget_controls(QtTailWidthSliderControl(self, layer))
        self._add_widget_controls(QtTailLengthSliderControl(self, layer))
        self._add_widget_controls(QtHeadLengthSliderControl(self, layer))
        self._add_widget_controls(QtTailDisplayCheckBoxControl(self, layer))
        self._add_widget_controls(QtIdCheckBoxControl(self, layer))
        self._add_widget_controls(QtGraphCheckBoxControl(self, layer))
