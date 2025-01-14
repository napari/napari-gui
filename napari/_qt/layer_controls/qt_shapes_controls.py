from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtFaceColorControl,
    QtTextVisibilityControl,
)
from napari._qt.layer_controls.widgets._shapes import (
    QtEdgeColorControl,
    QtEdgeWidthSliderControl,
)
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.shapes._shapes_constants import Mode
from napari.utils.action_manager import action_manager
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class QtShapesControls(QtLayerControls):
    """Qt view and controls for the napari Shapes layer.

    Parameters
    ----------
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.

    Attributes
    ----------
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        (SELECT, DIRECT, PAN_ZOOM, ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE,
        ADD_PATH, ADD_POLYGON, VERTEX_INSERT, VERTEX_REMOVE, TRANSFORM).
        Button group for shapes layer modes
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete selected shapes
    direct_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select individual vertices in shapes.
    ellipse_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add ellipses to shapes layer.
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.
    line_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add lines to shapes layer.
    move_back_button : qtpy.QtWidgets.QtModePushButton
        Button to move selected shape(s) to the back.
    move_front_button : qtpy.QtWidgets.QtModePushButton
        Button to move shape(s) to the front.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to pan/zoom shapes layer.
    path_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add paths to shapes layer.
    polygon_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add polygons to shapes layer.
    polygon_lasso_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add polygons to shapes layer with a lasso tool.
    qtEdgeColorControl.edgeColorEdit : qtpy.QtWidgets.QColorSwatchEdit
        ColorSwatchEdit controlling current edge color of the layer.
    qtEdgeColorControl.edgeColorLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge color chooser widget.
    qtEdgeWidthSliderControl.edgeWidthLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge width widget.
    qtEdgeWidthSliderControl.edgeWidthSlider : qtpy.QtWidgets.QSlider
        Slider controlling line edge width of layer.
    qtFaceColorControl.faceColorEdit : qtpy.QtWidgets.QSlider
        ColorSwatchEdit controlling current face color of the layer.
    qtFaceColorControl.faceColorLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current face color widget.
    qtOpacityBlendingControls.blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    qtOpacityBlendingControls.blendLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    qtOpacityBlendingControls.opacityLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    qtOpacityBlendingControls.opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    qtTextVisibilityControl.textDispCheckBox : qtpy.QtWidgets.QCheckbox
        Checkbox controlling if text on the layer is visible or not.
    qtTextVisibilityControl.textDispLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the text visibility widget.
    rectangle_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add rectangles to shapes layer.
    select_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select shapes.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform shapes layer.
    vertex_insert_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to insert vertex into shape.
    vertex_remove_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to remove vertex from shapes.

    Raises
    ------
    ValueError
        Raise error if shapes mode is not recognized.
    """

    layer: 'napari.layers.Shapes'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_shapes_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_shapes_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # Setup buttons
        self.select_button = self._radio_button(
            layer, 'select', Mode.SELECT, True, 'activate_select_mode'
        )
        self.direct_button = self._radio_button(
            layer, 'direct', Mode.DIRECT, True, 'activate_direct_mode'
        )
        self.rectangle_button = self._radio_button(
            layer,
            'rectangle',
            Mode.ADD_RECTANGLE,
            True,
            'activate_add_rectangle_mode',
        )
        self.ellipse_button = self._radio_button(
            layer,
            'ellipse',
            Mode.ADD_ELLIPSE,
            True,
            'activate_add_ellipse_mode',
        )
        self.line_button = self._radio_button(
            layer, 'line', Mode.ADD_LINE, True, 'activate_add_line_mode'
        )
        self.path_button = self._radio_button(
            layer, 'path', Mode.ADD_PATH, True, 'activate_add_path_mode'
        )
        self.polygon_button = self._radio_button(
            layer,
            'polygon',
            Mode.ADD_POLYGON,
            True,
            'activate_add_polygon_mode',
        )
        self.polygon_lasso_button = self._radio_button(
            layer,
            'polygon_lasso',
            Mode.ADD_POLYGON_LASSO,
            True,
            'activate_add_polygon_lasso_mode',
        )
        self.vertex_insert_button = self._radio_button(
            layer,
            'vertex_insert',
            Mode.VERTEX_INSERT,
            True,
            'activate_vertex_insert_mode',
        )
        self.vertex_remove_button = self._radio_button(
            layer,
            'vertex_remove',
            Mode.VERTEX_REMOVE,
            True,
            'activate_vertex_remove_mode',
        )

        self.move_front_button = QtModePushButton(
            layer,
            'move_front',
            slot=self.layer.move_to_front,
            tooltip=trans._('Move to front'),
        )
        action_manager.bind_button(
            'napari:move_shapes_selection_to_front', self.move_front_button
        )
        self.move_back_button = QtModePushButton(
            layer,
            'move_back',
            slot=self.layer.move_to_back,
            tooltip=trans._('Move to back'),
        )
        action_manager.bind_button(
            'napari:move_shapes_selection_to_back', self.move_back_button
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
            slot=self.layer.remove_selected,
            tooltip=trans._(
                'Delete selected shapes ({shortcut})',
                shortcut=Shortcut('Backspace').platform,
            ),
        )
        self._EDIT_BUTTONS += (
            self.delete_button,
            self.move_back_button,
            self.move_front_button,
        )
        self._on_editable_or_visible_change()

        self.button_grid.addWidget(self.vertex_remove_button, 0, 1)
        self.button_grid.addWidget(self.vertex_insert_button, 0, 2)
        self.button_grid.addWidget(self.delete_button, 0, 3)
        self.button_grid.addWidget(self.direct_button, 0, 4)
        self.button_grid.addWidget(self.select_button, 0, 5)
        self.button_grid.addWidget(self.move_back_button, 1, 0)
        self.button_grid.addWidget(self.move_front_button, 1, 1)
        self.button_grid.addWidget(self.ellipse_button, 1, 2)
        self.button_grid.addWidget(self.rectangle_button, 1, 3)
        self.button_grid.addWidget(self.polygon_button, 1, 4)
        self.button_grid.addWidget(self.polygon_lasso_button, 1, 5)
        self.button_grid.addWidget(self.line_button, 1, 6)
        self.button_grid.addWidget(self.path_button, 1, 7)
        self.button_grid.setContentsMargins(5, 0, 0, 5)
        self.button_grid.setColumnStretch(0, 1)
        self.button_grid.setSpacing(4)

        # Setup widgets controls
        self._add_widget_controls(QtEdgeWidthSliderControl(self, layer))
        self._add_widget_controls(
            QtEdgeColorControl(
                self,
                layer,
                tooltip=trans._(
                    'Click to set the edge color of currently selected shapes and any added afterwards'
                ),
            )
        )
        self._add_widget_controls(
            QtFaceColorControl(
                self,
                layer,
                tooltip=trans._(
                    'Click to set the face color of currently selected shapes and any added afterwards.'
                ),
            )
        )
        self._add_widget_controls(QtTextVisibilityControl(self, layer))

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when shapes layer mode changed.

        Available modes for shapes layer are:
        * SELECT
        * DIRECT
        * PAN_ZOOM
        * ADD_RECTANGLE
        * ADD_ELLIPSE
        * ADD_LINE
        * ADD_PATH
        * ADD_POLYGON
        * ADD_POLYGON_LASSO
        * VERTEX_INSERT
        * VERTEX_REMOVE
        * TRANSFORM

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not one of the available modes.
        """
        super()._on_mode_change(event)

    def _on_ndisplay_changed(self):
        self.layer.editable = self.ndisplay == 2
        super()._on_ndisplay_changed()
