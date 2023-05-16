import os
import warnings
from functools import wraps
from typing import TYPE_CHECKING

from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from napari._qt.dialogs.qt_modal import QtPopup
from napari._qt.widgets.qt_dims_sorter import QtDimsSorter
from napari._qt.widgets.qt_spinbox import QtSpinBox
from napari._qt.widgets.qt_tooltip import QtToolTipLabel
from napari.utils.action_manager import action_manager
from napari.utils.misc import in_ipython, in_jupyter, in_python_repl
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.viewer import ViewerModel


class QtLayerButtons(QFrame):
    """Button controls for napari layers.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    deleteButton : QtDeleteButton
        Button to delete selected layers.
    newLabelsButton : QtViewerPushButton
        Button to add new Label layer.
    newPointsButton : QtViewerPushButton
        Button to add new Points layer.
    newShapesButton : QtViewerPushButton
        Button to add new Shapes layer.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    def __init__(self, viewer: 'ViewerModel') -> None:
        super().__init__()

        self.viewer = viewer

        self.deleteButton = QtViewerPushButton(
            'delete_button', action='napari:delete_selected_layers'
        )

        self.newPointsButton = QtViewerPushButton(
            'new_points',
            trans._('New points layer'),
            lambda: self.viewer.add_points(
                ndim=max(self.viewer.dims.ndim, 2),
                scale=self.viewer.layers.extent.step,
            ),
        )

        self.newShapesButton = QtViewerPushButton(
            'new_shapes',
            trans._('New shapes layer'),
            lambda: self.viewer.add_shapes(
                ndim=max(self.viewer.dims.ndim, 2),
                scale=self.viewer.layers.extent.step,
            ),
        )
        self.newLabelsButton = QtViewerPushButton(
            'new_labels',
            trans._('New labels layer'),
            lambda: self.viewer._new_labels(),
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.newPointsButton)
        layout.addWidget(self.newShapesButton)
        layout.addWidget(self.newLabelsButton)
        layout.addStretch(0)
        layout.addWidget(self.deleteButton)
        self.setLayout(layout)

        # This can be always added after layergroups are fully capable
        if os.getenv("ALLOW_LAYERGROUPS"):
            self.newGroupButton = QtViewerPushButton(
                'new_group',
                trans._('New layer group'),
                self.viewer.layers.add_group,
            )
            layout.insertWidget(3, self.newGroupButton)


class QtViewerButtons(QFrame):
    """Button controls for the napari viewer.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    consoleButton : QtViewerPushButton
        Button to open iPython console within napari.
    rollDimsButton : QtViewerPushButton
        Button to roll orientation of spatial dimensions in the napari viewer.
    transposeDimsButton : QtViewerPushButton
        Button to transpose dimensions in the napari viewer.
    resetViewButton : QtViewerPushButton
        Button resetting the view of the rendered scene.
    gridViewButton : QtViewerPushButton
        Button to toggle grid view mode of layers on and off.
    ndisplayButton : QtViewerPushButton
        Button to toggle number of displayed dimensions.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    def __init__(self, viewer: 'ViewerModel') -> None:
        super().__init__()

        self.viewer = viewer

        self.consoleButton = QtViewerPushButton(
            'console', action='napari:toggle_console_visibility'
        )
        self.consoleButton.setProperty('expanded', False)
        if in_ipython() or in_jupyter() or in_python_repl():
            self.consoleButton.setEnabled(False)

        rdb = QtViewerPushButton('roll', action='napari:roll_axes')
        self.rollDimsButton = rdb
        rdb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        rdb.customContextMenuRequested.connect(self._open_roll_popup)

        self.transposeDimsButton = QtViewerPushButton(
            'transpose', action='napari:transpose_axes'
        )
        self.resetViewButton = QtViewerPushButton(
            'home', action='napari:reset_view'
        )
        gvb = QtViewerPushButton(
            'grid_view_button', action='napari:toggle_grid'
        )
        self.gridViewButton = gvb
        gvb.setCheckable(True)
        gvb.setChecked(viewer.grid.enabled)
        gvb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        gvb.customContextMenuRequested.connect(self._open_grid_popup)

        @self.viewer.grid.events.enabled.connect
        def _set_grid_mode_checkstate(event):
            gvb.setChecked(event.value)

        ndb = QtViewerPushButton(
            'ndisplay_button', action='napari:toggle_ndisplay'
        )
        self.ndisplayButton = ndb
        ndb.setCheckable(True)
        ndb.setChecked(self.viewer.dims.ndisplay == 3)
        ndb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        ndb.customContextMenuRequested.connect(self.open_perspective_popup)

        @self.viewer.dims.events.ndisplay.connect
        def _set_ndisplay_mode_checkstate(event):
            ndb.setChecked(event.value == 3)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.consoleButton)
        layout.addWidget(self.ndisplayButton)
        layout.addWidget(self.rollDimsButton)
        layout.addWidget(self.transposeDimsButton)
        layout.addWidget(self.gridViewButton)
        layout.addWidget(self.resetViewButton)
        layout.addStretch(0)
        self.setLayout(layout)

    def open_perspective_popup(self):
        """Show a slider to control the viewer `camera.perspective`."""
        if self.viewer.dims.ndisplay != 3:
            return

        # make slider connected to perspective parameter
        sld = QSlider(Qt.Orientation.Horizontal, self)
        sld.setRange(0, max(90, int(self.viewer.camera.perspective)))
        sld.setValue(self.viewer.camera.perspective)
        sld.valueChanged.connect(
            lambda v: setattr(self.viewer.camera, 'perspective', v)
        )

        # make layout
        layout = QHBoxLayout()
        layout.addWidget(QLabel(trans._('Perspective'), self))
        layout.addWidget(sld)

        # popup and show
        pop = QtPopup(self)
        pop.frame.setLayout(layout)
        pop.show_above_mouse()

    def _open_roll_popup(self):
        """Open a grid popup to manually order the dimensions"""
        if self.viewer.dims.ndisplay != 2:
            return

        dim_sorter = QtDimsSorter(self.viewer, self)
        dim_sorter.setObjectName('dim_sorter')

        # make layout
        layout = QHBoxLayout()
        layout.addWidget(dim_sorter)

        # popup and show
        pop = QtPopup(self)
        pop.frame.setLayout(layout)
        pop.show_above_mouse()

    def _open_grid_popup(self):
        """Open grid options pop up widget."""

        # widgets
        popup = QtPopup(self)
        grid_stride = QtSpinBox(popup)
        grid_width = QtSpinBox(popup)
        grid_height = QtSpinBox(popup)
        shape_help_symbol = QtToolTipLabel(self)
        stride_help_symbol = QtToolTipLabel(self)
        blank = QLabel(self)  # helps with placing help symbols.

        shape_help_msg = trans._(
            'Number of rows and columns in the grid. A value of -1 for either or both of width and height will trigger an auto calculation of the necessary grid shape to appropriately fill all the layers at the appropriate stride. 0 is not a valid entry.'
        )

        stride_help_msg = trans._(
            'Number of layers to place in each grid square before moving on to the next square. The default ordering is to place the most visible layer in the top left corner of the grid. A negative stride will cause the order in which the layers are placed in the grid to be reversed. 0 is not a valid entry.'
        )

        # set up
        stride_min = self.viewer.grid.__fields__['stride'].type_.ge
        stride_max = self.viewer.grid.__fields__['stride'].type_.le
        stride_not = self.viewer.grid.__fields__['stride'].type_.ne
        grid_stride.setObjectName("gridStrideBox")
        grid_stride.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_stride.setRange(stride_min, stride_max)
        grid_stride.setProhibitValue(stride_not)
        grid_stride.setValue(self.viewer.grid.stride)
        grid_stride.valueChanged.connect(self._update_grid_stride)
        self.grid_stride_box = grid_stride

        width_min = self.viewer.grid.__fields__['shape'].sub_fields[1].type_.ge
        width_not = self.viewer.grid.__fields__['shape'].sub_fields[1].type_.ne
        grid_width.setObjectName("gridWidthBox")
        grid_width.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_width.setMinimum(width_min)
        grid_width.setProhibitValue(width_not)
        grid_width.setValue(self.viewer.grid.shape[1])
        grid_width.valueChanged.connect(self._update_grid_width)
        self.grid_width_box = grid_width

        height_min = (
            self.viewer.grid.__fields__['shape'].sub_fields[0].type_.ge
        )
        height_not = (
            self.viewer.grid.__fields__['shape'].sub_fields[0].type_.ne
        )
        grid_height.setObjectName("gridStrideBox")
        grid_height.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_height.setMinimum(height_min)
        grid_height.setProhibitValue(height_not)
        grid_height.setValue(self.viewer.grid.shape[0])
        grid_height.valueChanged.connect(self._update_grid_height)
        self.grid_height_box = grid_height

        shape_help_symbol.setObjectName("help_label")
        shape_help_symbol.setToolTip(shape_help_msg)

        stride_help_symbol.setObjectName("help_label")
        stride_help_symbol.setToolTip(stride_help_msg)

        # layout
        form_layout = QFormLayout()
        form_layout.insertRow(0, QLabel(trans._('Grid stride:')), grid_stride)
        form_layout.insertRow(1, QLabel(trans._('Grid width:')), grid_width)
        form_layout.insertRow(2, QLabel(trans._('Grid height:')), grid_height)

        help_layout = QVBoxLayout()
        help_layout.addWidget(stride_help_symbol)
        help_layout.addWidget(blank)
        help_layout.addWidget(shape_help_symbol)

        layout = QHBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(help_layout)

        popup.frame.setLayout(layout)

        popup.show_above_mouse()

        # adjust placement of shape help symbol.  Must be done last
        # in order for this movement to happen.
        delta_x = 0
        delta_y = -15
        shape_pos = (
            shape_help_symbol.x() + delta_x,
            shape_help_symbol.y() + delta_y,
        )
        shape_help_symbol.move(QPoint(*shape_pos))

    def _update_grid_width(self, value):
        """Update the width value in grid shape.

        Parameters
        ----------
        value : int
            New grid width value.
        """

        self.viewer.grid.shape = (self.viewer.grid.shape[0], value)

    def _update_grid_stride(self, value):
        """Update stride in grid settings.

        Parameters
        ----------
        value : int
            New grid stride value.
        """

        self.viewer.grid.stride = value

    def _update_grid_height(self, value):
        """Update height value in grid shape.

        Parameters
        ----------
        value : int
            New grid height value.
        """

        self.viewer.grid.shape = (value, self.viewer.grid.shape[1])


def _omit_viewer_args(constructor):
    @wraps(constructor)
    def _func(*args, **kwargs):
        if len(args) > 1 and not isinstance(args[1], str):
            warnings.warn(
                trans._(
                    "viewer argument is deprecated since 0.4.14 and should not be used"
                ),
                category=FutureWarning,
                stacklevel=2,
            )
            args = args[:1] + args[2:]
        if "viewer" in kwargs:
            warnings.warn(
                trans._(
                    "viewer argument is deprecated since 0.4.14 and should not be used"
                ),
                category=FutureWarning,
                stacklevel=2,
            )
            del kwargs["viewer"]
        return constructor(*args, **kwargs)

    return _func


class QtViewerPushButton(QPushButton):
    """Push button.

    Parameters
    ----------
    button_name : str
        Name of button.
    tooltip : str
        Tooltip for button. If empty then `button_name` is used
    slot : Callable, optional
        callable to be triggered on button click
    action : str
        action name to be triggered on button click
    """

    @_omit_viewer_args
    def __init__(
        self, button_name: str, tooltip: str = '', slot=None, action: str = ''
    ) -> None:
        super().__init__()

        self.setToolTip(tooltip or button_name)
        self.setProperty('mode', button_name)
        if slot is not None:
            self.clicked.connect(slot)
        if action:
            action_manager.bind_button(action, self)
