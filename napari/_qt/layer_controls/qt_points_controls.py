from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtFaceColorControl,
    QtOutSliceCheckBoxControl,
    QtTextVisibilityControl,
)
from napari._qt.layer_controls.widgets._points import (
    QtBorderColorControl,
    QtCurrentSizeSliderControl,
    QtSymbolComboBoxControl,
)
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.points._points_constants import Mode
from napari.utils.action_manager import action_manager
from napari.utils.events import disconnect_events
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class QtPointsControls(QtLayerControls):
    """Qt view and controls for the napari Points layer.

    Parameters
    ----------
    layer : napari.layers.Points
        An instance of a napari Points layer.

    Attributes
    ----------
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    addition_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add points to layer.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of points layer modes (ADD, PAN_ZOOM, SELECT).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete points from layer.
    layer : napari.layers.Points
        An instance of a napari Points layer.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button for pan/zoom mode.
    qtBorderColorControl.borderColorEdit : napari._qt.widgets.qt_color_swatch.QColorSwatchEdit
        Widget to select display color for points borders.
    qtBorderColorControl.borderColorEditLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current egde color chooser widget.
    qtCurrentSizeSliderControl.sizeSlider : napari._qt.widgets._slider_compat.QSlider
        Slider controlling size of points.
    qtCurrentSizeSliderControl.sizeSliderLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the points size chooser widget.
    qtFaceColorControl.faceColorEdit : napari._qt.widgets.qt_color_swatch.QColorSwatchEdit
        Widget to select display color for points faces.
    qtFaceColorControl.faceColorLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current face color chooser widget.
    qtOpacityBlendingControls.blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    qtOpacityBlendingControls.blendLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    qtOpacityBlendingControls.opacityLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    qtOpacityBlendingControls.opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    qtOutSliceCheckBoxControl.outOfSliceCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether to render out of slice.
    qtOutSliceCheckBoxControl.outOfSliceCheckBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the out of slice display enablement chooser widget.
    qtSymbolComboBoxControl.symbolComboBox : qtpy.QtWidgets.QComboBox
        Dropdown list of symbol options for points markers.
    qtSymbolComboBoxControl.symbolComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for points symbol chooser widget.
    qtTextVisibilityControl.textDispCheckBox : qtpy.QtWidgets.QCheckbox
        Checkbox controlling if text on the layer is visible or not.
    qtTextVisibilityControl.textDispLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the text visibility widget.
    select_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select points from layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select transform mode.

    Raises
    ------
    ValueError
        Raise error if points mode is not recognized.
        Points mode must be one of: ADD, PAN_ZOOM, or SELECT.
    """

    layer: 'napari.layers.Points'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_points_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_points_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # Setup buttons
        self.select_button = self._radio_button(
            layer,
            'select_points',
            Mode.SELECT,
            True,
            'activate_points_select_mode',
        )
        self.addition_button = self._radio_button(
            layer,
            'add_points',
            Mode.ADD,
            True,
            'activate_points_add_mode',
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
        )
        action_manager.bind_button(
            'napari:delete_selected_points', self.delete_button
        )
        self._EDIT_BUTTONS += (self.delete_button,)
        self._on_editable_or_visible_change()

        self.button_grid.addWidget(self.delete_button, 0, 3)
        self.button_grid.addWidget(self.addition_button, 0, 4)
        self.button_grid.addWidget(self.select_button, 0, 5)

        # Setup widgets controls
        self._add_widget_controls(QtCurrentSizeSliderControl(self, layer))
        self._add_widget_controls(QtSymbolComboBoxControl(self, layer))
        self._add_widget_controls(
            QtFaceColorControl(
                self,
                layer,
                tooltip=trans._(
                    'Click to set the face color of currently selected points and any added afterwards.'
                ),
            )
        )
        self._add_widget_controls(QtBorderColorControl(self, layer))
        self._add_widget_controls(QtTextVisibilityControl(self, layer))
        self._add_widget_controls(QtOutSliceCheckBoxControl(self, layer))

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when points layer mode is changed.

        Available modes for points layer are:
        * ADD
        * SELECT
        * PAN_ZOOM
        * TRANSFORM

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not ADD, PAN_ZOOM, TRANSFORM or SELECT.
        """
        super()._on_mode_change(event)

    def _on_ndisplay_changed(self):
        self.layer.editable = not (self.layer.ndim == 2 and self.ndisplay == 3)
        super()._on_ndisplay_changed()

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
