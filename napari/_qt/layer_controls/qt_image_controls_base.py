from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtAutoScaleControl,
    QtColormapControl,
    QtContrastLimitsSliderControl,
    QtGammaSliderControl,
)

if TYPE_CHECKING:
    from napari.layers import Image


class QtBaseImageControls(QtLayerControls):
    """Superclass for classes requiring colormaps, contrast & gamma sliders.

    This class is never directly instantiated anywhere.
    It is subclassed by QtImageControls and QtSurfaceControls.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    layer : napari.layers.Layer
        An instance of a napari layer.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to pan/zoom shapes layer.
    qtAutoScaleControl.autoScaleBar : qtpy.QtWidgets.QWidget
        Widget to wrap push buttons related with the layer auto-contrast funtionality.
    qtColormapControl.colormapWidget : qtpy.QtWidgets.QWidget
        Widget to wrap combobox and label widgets related with the layer colormap attribute.
    qtContrastLimitsSliderControl.contrastLimitsSlider : superqt.QRangeSlider
        Contrast range slider widget.
    qtGammaSliderControl.gammaSlider : qtpy.QtWidgets.QSlider
        Gamma adjustment slider widget.
    qtOpacityBlendingControls.blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    qtOpacityBlendingControls.blendLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    qtOpacityBlendingControls.opacityLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    qtOpacityBlendingControls.opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform shapes layer.
    """

    def __init__(self, layer: Image) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._add_widget_controls(QtContrastLimitsSliderControl(self, layer))
        self._add_widget_controls(QtAutoScaleControl(self, layer))
        self._add_widget_controls(QtGammaSliderControl(self, layer))
        self._add_widget_controls(QtColormapControl(self, layer))
