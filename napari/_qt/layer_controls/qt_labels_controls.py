from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets._labels import (
    QtBrushSizeSliderControl,
    QtColorModeComboBoxControl,
    QtContiguousCheckBoxControl,
    QtContourSpinBoxControl,
    QtDisplaySelectedLabelCheckBoxControl,
    QtLabelControl,
    QtLabelRenderControl,
    QtNdimSpinBoxControl,
    QtPreserveLabelsCheckBoxControl,
)
from napari._qt.utils import set_widgets_enabled_with_opacity
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.labels._labels_constants import Mode
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


INT32_MAX = 2**31 - 1


class QtLabelsControls(QtLayerControls):
    """Qt view and controls for the napari Labels layer.

    Parameters
    ----------
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of labels layer modes: PAN_ZOOM, PICKER, PAINT, ERASE, or
        FILL.
    colormapUpdate : qtpy.QtWidgets.QPushButton
        Button to update colormap of label layer.
    erase_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select ERASE mode on Labels layer.
    fill_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select FILL mode on Labels layer.
    layer : napari.layers.Labels
        An instance of a napari Labels layer.
    paint_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select PAINT mode on Labels layer.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select PAN_ZOOM mode on Labels layer.
    pick_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select PICKER mode on Labels layer.
    qtBrushSizeSliderControl.brushSizeSlider : qtpy.QtWidgets.QSlider
        Slider controlling current brush size of the layer.
    qtBrushSizeSliderControl.brushSizeSliderLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the brush size chooser widget.
    qtColorModeComboBoxControl.colorModeComboBox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current color mode of the layer.
    qtColorModeComboBoxControl.colorModeComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the color mode chooser widget.
    qtContiguousCheckBoxControl.contigCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if label layer is contiguous.
    qtContiguousCheckBoxControl.contigCheckBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the contiguous model chooser widget.
    qtContourSpinBoxControl.contourSpinBox : superqt.QLargeSpinBox
        Spinbox to control the layer contour thickness.
    qtContourSpinBoxControl.contourSpinBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer contour thickness chooser widget.
    qtDisplaySelectedLabelCheckBoxControl.selectedColorCheckbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if only currently selected label is shown.
    qtDisplaySelectedLabelCheckBoxControl.selectedColorCheckboxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer should show only currently selected label chooser widget.
    qtLabelControl.labelColor : qtpy.QtWidget.QWidget
        Wrapper widget for the selectionSpinBox and colorBox widgets.
    qtLabelControl.labelColorLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the label chooser widget.
    qtLabelRenderControl.isoGradientComboBox : superqt.QEnumComboBox
        Combobox to control gradient method when isosurface rendering is selected.
    qtLabelRenderControl.isoGradientComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the chooser widget of the gradient to use when labels are using isosurface rendering.
    qtLabelRenderControl.renderComboBox : superqt.QEnumComboBox
        Combobox to control current label render method.
    qtLabelRenderControl.renderComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the way current label should be rendered chooser widget.
    qtNdimSpinBoxControl.ndimSpinBox : qtpy.QtWidgets.QSpinBox
        Spinbox to control the number of editable dimensions of label layer.
    qtNdimSpinBoxControl.ndimSpinBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the number of editable dimensions chooser widget.
    qtOpacityBlendingControls.blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    qtOpacityBlendingControls.blendLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    qtOpacityBlendingControls.opacityLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    qtOpacityBlendingControls.opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    qtPreserveLabelsCheckBoxControl.preserveLabelsCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if existing labels are preserved.
    qtPreserveLabelsCheckBoxControl.preserveLabelsCheckBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer should preserve labels chooser widget.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select TRANSFORM mode on Labels layer.

    Raises
    ------
    ValueError
        Raise error if label mode is not PAN_ZOOM, PICKER, PAINT, ERASE, or
        FILL.
    """

    layer: 'napari.layers.Labels'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_labels_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_labels_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # Setup buttons
        # shuffle colormap button
        self.colormapUpdate = QtModePushButton(
            layer,
            'shuffle',
            slot=self.changeColor,
            tooltip=trans._('Shuffle colors'),
        )

        self.pick_button = self._radio_button(
            layer,
            'picker',
            Mode.PICK,
            True,
            'activate_labels_picker_mode',
        )
        self.paint_button = self._radio_button(
            layer,
            'paint',
            Mode.PAINT,
            True,
            'activate_labels_paint_mode',
        )
        self.polygon_button = self._radio_button(
            layer,
            'labels_polygon',
            Mode.POLYGON,
            True,
            'activate_labels_polygon_mode',
        )
        self.fill_button = self._radio_button(
            layer,
            'fill',
            Mode.FILL,
            True,
            'activate_labels_fill_mode',
        )
        self.erase_button = self._radio_button(
            layer,
            'erase',
            Mode.ERASE,
            True,
            'activate_labels_erase_mode',
        )
        # don't bind with action manager as this would remove "Toggle with {shortcut}"
        self._on_editable_or_visible_change()

        self.button_grid.addWidget(self.colormapUpdate, 0, 0)
        self.button_grid.addWidget(self.erase_button, 0, 1)
        self.button_grid.addWidget(self.paint_button, 0, 2)
        self.button_grid.addWidget(self.polygon_button, 0, 3)
        self.button_grid.addWidget(self.fill_button, 0, 4)
        self.button_grid.addWidget(self.pick_button, 0, 5)

        # Setup widgets controls
        self._add_widget_controls(QtLabelControl(self, layer))
        self._add_widget_controls(QtBrushSizeSliderControl(self, layer))
        self._renderControl = QtLabelRenderControl(self, layer)
        self._add_widget_controls(self._renderControl)
        self._add_widget_controls(QtColorModeComboBoxControl(self, layer))
        self._add_widget_controls(QtContourSpinBoxControl(self, layer))
        self._add_widget_controls(QtNdimSpinBoxControl(self, layer))
        self._add_widget_controls(QtContiguousCheckBoxControl(self, layer))
        self._add_widget_controls(QtPreserveLabelsCheckBoxControl(self, layer))
        self._add_widget_controls(
            QtDisplaySelectedLabelCheckBoxControl(self, layer)
        )

        self._on_ndisplay_changed()

    def _on_mode_change(self, event):
        """Receive layer model mode change event and update checkbox ticks.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not PAN_ZOOM, PICK, PAINT, ERASE, FILL
            or TRANSFORM
        """
        super()._on_mode_change(event)

    def changeColor(self):
        """Change colormap of the label layer."""
        self.layer.new_colormap()

    def _on_editable_or_visible_change(self):
        super()._on_editable_or_visible_change()
        self._set_polygon_tool_state()

    def _on_ndisplay_changed(self):
        show_3d_widgets = self.ndisplay == 3
        if show_3d_widgets:
            self._renderControl._on_display_change_show()
        else:
            self._renderControl._on_display_change_hide()
        self._on_editable_or_visible_change()
        self._set_polygon_tool_state()
        super()._on_ndisplay_changed()

    def _set_polygon_tool_state(self):
        if hasattr(self, 'polygon_button'):
            set_widgets_enabled_with_opacity(
                self, [self.polygon_button], self._is_polygon_tool_enabled()
            )

    def _is_polygon_tool_enabled(self):
        return (
            self.layer.editable
            and self.layer.visible
            and self.layer.n_edit_dimensions == 2
            and self.ndisplay == 2
        )
