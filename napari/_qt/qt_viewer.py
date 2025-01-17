from __future__ import annotations

import logging
import sys
import traceback
import warnings
import weakref
from collections.abc import Sequence
from pathlib import Path
from types import FrameType
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)
from weakref import WeakSet, ref

import numpy as np
from qtpy.QtCore import QCoreApplication, QObject, Qt, QUrl
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QFileDialog, QSplitter, QVBoxLayout, QWidget
from superqt import ensure_main_thread

from napari._qt.containers import QtLayerList
from napari._qt.dialogs.qt_reader_dialog import handle_gui_reading
from napari._qt.dialogs.screenshot_dialog import ScreenshotDialog
from napari._qt.perf.qt_performance import QtPerformance
from napari._qt.utils import QImg2array
from napari._qt.widgets.qt_dims import QtDims
from napari._qt.widgets.qt_viewer_buttons import (
    QtLayerButtons,
    QtViewerButtons,
)
from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
from napari._qt.widgets.qt_welcome import QtWidgetOverlay
from napari.components.camera import Camera
from napari.components.layerlist import LayerList
from napari.errors import MultipleReaderError, ReaderPluginError
from napari.layers.base.base import Layer
from napari.plugins import _npe2
from napari.settings import get_settings
from napari.settings._application import DaskSettings
from napari.utils import config, perf, resize_dask_cache
from napari.utils.action_manager import action_manager
from napari.utils.history import (
    get_open_history,
    get_save_history,
    update_open_history,
    update_save_history,
)
from napari.utils.io import imsave
from napari.utils.key_bindings import KeymapHandler
from napari.utils.misc import in_ipython, in_jupyter
from napari.utils.naming import CallerFrame
from napari.utils.notifications import show_info
from napari.utils.translations import trans
from napari_builtins.io import imsave_extensions

from napari._vispy import VispyCanvas, create_vispy_layer  # isort:skip

if TYPE_CHECKING:
    from napari_console import QtConsole
    from npe2.manifest.contributions import WriterContribution

    from napari._qt.layer_controls import QtLayerControlsContainer
    from napari.components import ViewerModel
    from napari.utils.events import Event


def _npe2_decode_selected_filter(
    ext_str: str, selected_filter: str, writers: Sequence[WriterContribution]
) -> Optional[WriterContribution]:
    """Determine the writer that should be invoked to save data.

    When npe2 can be imported, resolves a selected file extension
    string into a specific writer. Otherwise, returns None.
    """
    # When npe2 is not present, `writers` is expected to be an empty list,
    # `[]`. This function will return None.

    for entry, writer in zip(
        ext_str.split(';;'),
        writers,
    ):
        if entry.startswith(selected_filter):
            return writer
    return None


def _extension_string_for_layers(
    layers: Sequence[Layer],
) -> tuple[str, list[WriterContribution]]:
    """Return an extension string and the list of corresponding writers.

    The extension string is a ";;" delimeted string of entries. Each entry
    has a brief description of the file type and a list of extensions.

    The writers, when provided, are the npe2.manifest.io.WriterContribution
    objects. There is one writer per entry in the extension string. If npe2
    is not importable, the list of writers will be empty.
    """
    # try to use npe2
    ext_str, writers = _npe2.file_extensions_string_for_layers(layers)
    if ext_str:
        return ext_str, writers

    # fallback to old behavior

    if len(layers) == 1:
        selected_layer = layers[0]
        # single selected layer.
        if selected_layer._type_string == 'image':
            ext = imsave_extensions()

            ext_list = [f'*{val}' for val in ext]
            ext_str = ';;'.join(ext_list)

            ext_str = trans._(
                'All Files (*);; Image file types:;;{ext_str}',
                ext_str=ext_str,
            )

        elif selected_layer._type_string == 'points':
            ext_str = trans._('All Files (*);; *.csv;;')

        else:
            # layer other than image or points
            ext_str = trans._('All Files (*);;')

    else:
        # multiple layers.
        ext_str = trans._('All Files (*);;')
    return ext_str, []


class QtViewer(QSplitter):
    """Qt view for the napari Viewer model.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    show_welcome_screen : bool, optional
        Flag to show a welcome message when no layers are present in the
        canvas. Default is `False`.
    canvas_class : napari._vispy.canvas.VispyCanvas
        The VispyCanvas class providing the Vispy SceneCanvas. Users can also
        have a custom canvas here.

    Attributes
    ----------
    canvas : napari._vispy.canvas.VispyCanvas
        The VispyCanvas class providing the Vispy SceneCanvas. Users can also
        have a custom canvas here.
    dims : napari.qt_dims.QtDims
        Dimension sliders; Qt View for Dims model.
    show_welcome_screen : bool
        Boolean indicating whether to show the welcome screen.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    _key_map_handler : napari.utils.key_bindings.KeymapHandler
        KeymapHandler handling the calling functionality when keys are pressed that have a callback function mapped
    _qt_poll : Optional[napari._qt.experimental.qt_poll.QtPoll]
        A QtPoll object required for the monitor.
    _remote_manager : napari.components.experimental.remote.RemoteManager
        A remote manager processing commands from remote clients and sending out messages when polled.
    _welcome_widget : napari._qt.widgets.qt_welcome.QtWidgetOverlay
        QtWidgetOverlay providing the stacked widgets for the welcome page.
    """

    _instances = WeakSet()

    def __init__(
        self,
        viewer: ViewerModel,
        show_welcome_screen: bool = False,
        canvas_class: type[VispyCanvas] = VispyCanvas,
    ) -> None:
        super().__init__()
        self._instances.add(self)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._show_welcome_screen = show_welcome_screen

        QCoreApplication.setAttribute(
            Qt.AA_UseStyleSheetPropagationInWidgetStyles, True
        )

        self.viewer = viewer
        self.dims = QtDims(self.viewer.dims)
        self._controls = None
        self._layers = None
        self._layersButtons = None
        self._viewerButtons = None
        self._key_map_handler = KeymapHandler()
        self._key_map_handler.keymap_providers = [self.viewer]
        self._console_backlog = []
        self._console = None

        self._dockLayerList = None
        self._dockLayerControls = None
        self._dockConsole = None
        self._dockPerformance = None

        # This dictionary holds the corresponding vispy visual for each layer
        self.canvas = canvas_class(
            viewer=viewer,
            parent=self,
            key_map_handler=self._key_map_handler,
            size=self.viewer._canvas_size,
            autoswap=get_settings().experimental.autoswap_buffers,  # see #5734
        )

        # Stacked widget to provide a welcome page
        self._welcome_widget = QtWidgetOverlay(self, self.canvas.native)
        self._welcome_widget.set_welcome_visible(show_welcome_screen)
        self._welcome_widget.sig_dropped.connect(self.dropEvent)
        self._welcome_widget.leave.connect(self._leave_canvas)
        self._welcome_widget.enter.connect(self._enter_canvas)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 2, 0, 2)
        main_layout.addWidget(self._welcome_widget)
        main_layout.addWidget(self.dims)
        main_layout.setSpacing(0)
        main_widget.setLayout(main_layout)

        self.setOrientation(Qt.Orientation.Vertical)
        self.addWidget(main_widget)

        self.viewer._layer_slicer.events.ready.connect(self._on_slice_ready)

        self._on_active_change()
        self.viewer.layers.events.inserted.connect(self._update_camera_depth)
        self.viewer.layers.events.removed.connect(self._update_camera_depth)
        self.viewer.dims.events.ndisplay.connect(self._update_camera_depth)
        self.viewer.layers.events.inserted.connect(self._update_welcome_screen)
        self.viewer.layers.events.removed.connect(self._update_welcome_screen)
        self.viewer.layers.selection.events.active.connect(
            self._on_active_change
        )

        self.viewer.layers.events.inserted.connect(self._on_add_layer_change)

        self.setAcceptDrops(True)

        # Create the experimental QtPool for the monitor.
        self._qt_poll = _create_qt_poll(self, self.viewer.camera)

        # Create the experimental RemoteManager for the monitor.
        self._remote_manager = _create_remote_manager(
            self.viewer.layers, self._qt_poll
        )

        # bind shortcuts stored in settings last.
        self._bind_shortcuts()

        settings = get_settings()
        self._update_dask_cache_settings(settings.application.dask)

        settings.application.dask.events.connect(
            self._update_dask_cache_settings
        )

        for layer in self.viewer.layers:
            self._add_layer(layer)

    @property
    def view(self):
        """
        Rectangular  vispy viewbox widget in which a subscene is rendered. Access directly within the QtViewer will
        become deprecated.
        """
        warnings.warn(
            trans._(
                'Access to QtViewer.view is deprecated since 0.5.0 and will be removed in the napari 0.6.0. Change to QtViewer.canvas.view instead.'
            ),
            FutureWarning,
            stacklevel=2,
        )
        return self.canvas.view

    @property
    def camera(self):
        """
        The Vispy camera class which contains both the 2d and 3d camera used to describe the perspective by which a
        scene is viewed and interacted with. Access directly within the QtViewer will become deprecated.
        """
        warnings.warn(
            trans._(
                'Access to QtViewer.camera will become deprecated in the 0.6.0. Change to QtViewer.canvas.camera instead.'
            ),
            FutureWarning,
            stacklevel=2,
        )
        return self.canvas.camera

    @property
    def chunk_receiver(self) -> None:
        warnings.warn(
            trans._(
                'QtViewer.chunk_receiver is deprecated in version 0.5 and will be removed in a later version. '
                'More generally the old approach to async loading was removed in version 0.5 so this value is always None. '
                'If you need to specifically use the old approach, continue to use the latest 0.4 release.'
            ),
            DeprecationWarning,
        )

    @staticmethod
    def _update_dask_cache_settings(
        dask_setting: Union[DaskSettings, Event] = None,
    ):
        """Update dask cache to match settings."""
        if not dask_setting:
            return
        if not isinstance(dask_setting, DaskSettings):
            dask_setting = get_settings().application.dask

        enabled = dask_setting.enabled
        size = dask_setting.cache
        resize_dask_cache(int(int(enabled) * size * 1e9))

    @property
    def controls(self) -> QtLayerControlsContainer:
        """Qt view for GUI controls."""
        if self._controls is None:
            # Avoid circular import.
            from napari._qt.layer_controls import QtLayerControlsContainer

            self._controls = QtLayerControlsContainer(self.viewer)
        return self._controls

    @property
    def layers(self) -> QtLayerList:
        """Qt view for LayerList controls."""
        if self._layers is None:
            self._layers = QtLayerList(self.viewer.layers)
        return self._layers

    @property
    def layerButtons(self) -> QtLayerButtons:
        """Button controls for napari layers."""
        if self._layersButtons is None:
            self._layersButtons = QtLayerButtons(self.viewer)
        return self._layersButtons

    @property
    def viewerButtons(self) -> QtViewerButtons:
        """Button controls for the napari viewer."""
        if self._viewerButtons is None:
            self._viewerButtons = QtViewerButtons(self.viewer)
        return self._viewerButtons

    @property
    def dockLayerList(self) -> QtViewerDockWidget:
        """QWidget wrapped in a QDockWidget with forwarded viewer events."""
        if self._dockLayerList is None:
            layerList = QWidget()
            layerList.setObjectName('layerList')
            layerListLayout = QVBoxLayout()
            layerListLayout.addWidget(self.layerButtons)
            layerListLayout.addWidget(self.layers)
            layerListLayout.addWidget(self.viewerButtons)
            layerListLayout.setContentsMargins(8, 4, 8, 6)
            layerList.setLayout(layerListLayout)
            self._dockLayerList = QtViewerDockWidget(
                self,
                layerList,
                name=trans._('layer list'),
                area='left',
                allowed_areas=['left', 'right'],
                object_name='layer list',
                close_btn=False,
            )
        return self._dockLayerList

    @property
    def dockLayerControls(self) -> QtViewerDockWidget:
        """QWidget wrapped in a QDockWidget with forwarded viewer events."""
        if self._dockLayerControls is None:
            self._dockLayerControls = QtViewerDockWidget(
                self,
                self.controls,
                name=trans._('layer controls'),
                area='left',
                allowed_areas=['left', 'right'],
                object_name='layer controls',
                close_btn=False,
            )
        return self._dockLayerControls

    @property
    def dockConsole(self) -> QtViewerDockWidget:
        """QWidget wrapped in a QDockWidget with forwarded viewer events."""
        if self._dockConsole is None:
            self._dockConsole = QtViewerDockWidget(
                self,
                QWidget(),
                name=trans._('console'),
                area='bottom',
                allowed_areas=['top', 'bottom'],
                object_name='console',
                close_btn=False,
            )
            self._dockConsole.setVisible(False)
            self._dockConsole.visibilityChanged.connect(self._ensure_connect)
        return self._dockConsole

    @property
    def dockPerformance(self) -> QtViewerDockWidget:
        if self._dockPerformance is None:
            self._dockPerformance = self._create_performance_dock_widget()
        return self._dockPerformance

    @property
    def layer_to_visual(self):
        """Mapping of Napari layer to Vispy layer. Added for backward compatibility"""
        return self.canvas.layer_to_visual

    def _leave_canvas(self):
        """disable status on canvas leave"""
        self.viewer.status = ''
        self.viewer.mouse_over_canvas = False

    def _enter_canvas(self):
        """enable status on canvas enter"""
        self.viewer.status = 'Ready'
        self.viewer.mouse_over_canvas = True

    def _ensure_connect(self):
        # lazy load console
        id(self.console)

    def _bind_shortcuts(self):
        """Bind shortcuts stored in SETTINGS to actions."""
        for action, shortcuts in get_settings().shortcuts.shortcuts.items():
            action_manager.unbind_shortcut(action)
            for shortcut in shortcuts:
                action_manager.bind_shortcut(action, shortcut)

    def _create_performance_dock_widget(self):
        """Create the dock widget that shows performance metrics."""
        if perf.perf_config is not None:
            return QtViewerDockWidget(
                self,
                QtPerformance(),
                name=trans._('performance'),
                area='bottom',
            )
        return None

    def _weakref_if_possible(self, obj):
        """Create a weakref to obj.

        Parameters
        ----------
        obj : object
            Cannot create weakrefs to many Python built-in datatypes such as
            list, dict, str.

            From https://docs.python.org/3/library/weakref.html: "Objects which
            support weak references include class instances, functions written
            in Python (but not in C), instance methods, sets, frozensets, some
            file objects, generators, type objects, sockets, arrays, deques,
            regular expression pattern objects, and code objects."

        Returns
        -------
        weakref or object
            Returns a weakref if possible.
        """
        try:
            newref = ref(obj)
        except TypeError:
            newref = obj
        return newref

    def _unwrap_if_weakref(self, value):
        """Return value or if that is weakref the object referenced by value.

        Parameters
        ----------
        value : object or weakref
            No-op for types other than weakref.

        Returns
        -------
        unwrapped: object or None
            Returns referenced object, or None if weakref is dead.
        """
        unwrapped = value() if isinstance(value, ref) else value
        return unwrapped

    def add_to_console_backlog(self, variables):
        """Save variables for pushing to console when it is instantiated.

        This function will create weakrefs when possible to avoid holding on to
        too much memory unnecessarily.

        Parameters
        ----------
        variables : dict, str or list/tuple of str
            The variables to inject into the console's namespace. If a dict, a
            simple update is done. If a str, the string is assumed to have
            variable names separated by spaces. A list/tuple of str can also
            be used to give the variable names. If just the variable names are
            give (list/tuple/str) then the variable values looked up in the
            callers frame.
        """
        if isinstance(variables, (str, list, tuple)):
            if isinstance(variables, str):
                vlist = variables.split()
            else:
                vlist = variables
            vdict = {}
            cf = sys._getframe(2)
            for name in vlist:
                try:
                    vdict[name] = eval(name, cf.f_globals, cf.f_locals)
                except NameError:
                    logging.warning(
                        'Could not get variable %s from %s',
                        name,
                        cf.f_code.co_name,
                    )
        elif isinstance(variables, dict):
            vdict = variables
        else:
            raise TypeError('variables must be a dict/str/list/tuple')
        # weakly reference values if possible
        new_dict = {k: self._weakref_if_possible(v) for k, v in vdict.items()}
        self.console_backlog.append(new_dict)

    @property
    def console_backlog(self):
        """List: items to push to console when instantiated."""
        return self._console_backlog

    def _get_console(self) -> Optional[QtConsole]:
        """Function to setup console.

        Returns
        -------
        console : QtConsole or None
            The napari console.

        Notes
        _____
        _get_console extracted to separate function to simplify testing.

        """
        try:
            import numpy as np

            # QtConsole imports debugpy that overwrites default breakpoint.
            # It makes problems with debugging if you do not know this.
            # So we do not want to overwrite it if it is already set.
            breakpoint_handler = sys.breakpointhook
            from napari_console import QtConsole

            sys.breakpointhook = breakpoint_handler

            import napari

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                console = QtConsole(self.viewer, style_sheet=self.styleSheet())
                console.push(
                    {'napari': napari, 'action_manager': action_manager}
                )
                with CallerFrame(_in_napari) as c:
                    if c.frame.f_globals.get('__name__', '') == '__main__':
                        console.push({'np': np})
                for i in self.console_backlog:
                    # recover weak refs
                    console.push(
                        {
                            k: self._unwrap_if_weakref(v)
                            for k, v in i.items()
                            if self._unwrap_if_weakref(v) is not None
                        }
                    )
                return console
        except ModuleNotFoundError:
            warnings.warn(
                trans._(
                    'napari-console not found. It can be installed with'
                    ' "pip install napari_console"'
                ),
                stacklevel=1,
            )
            return None
        except ImportError:
            traceback.print_exc()
            warnings.warn(
                trans._(
                    'error importing napari-console. See console for full error.'
                ),
                stacklevel=1,
            )
            return None

    @property
    def console(self):
        """QtConsole: iPython console terminal integrated into the napari GUI."""
        if self._console is None:
            self.console = self._get_console()
            self._console_backlog = []
        return self._console

    @console.setter
    def console(self, console):
        self._console = console
        if console is not None:
            self.dockConsole.setWidget(console)
            console.setParent(self.dockConsole)

    @ensure_main_thread
    def _on_slice_ready(self, event):
        """Callback connected to `viewer._layer_slicer.events.ready`.

        Provides updates after slicing using the slice response data.
        This only gets triggered on the async slicing path.
        """
        responses: dict[weakref.ReferenceType[Layer], Any] = event.value
        logging.debug('QtViewer._on_slice_ready: %s', responses)
        for weak_layer, response in responses.items():
            if layer := weak_layer():
                # Update the layer slice state to temporarily support behavior
                # that depends on it.
                layer._update_slice_response(response)
                # Update the layer's loaded state before everything else,
                # because they may rely on its updated value.
                layer._update_loaded_slice_id(response.request_id)
                # The rest of `Layer.refresh` after `set_view_slice`, where
                # `set_data` notifies the corresponding vispy layer of the new
                # slice.
                layer.events.set_data()
                layer._update_thumbnail()
                layer._set_highlight(force=True)

    def _on_active_change(self):
        """When active layer changes change keymap handler."""
        self._key_map_handler.keymap_providers = (
            [self.viewer]
            if self.viewer.layers.selection.active is None
            else [self.viewer.layers.selection.active, self.viewer]
        )

    def _on_add_layer_change(self, event):
        """When a layer is added, set its parent and order.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        layer = event.value
        self._add_layer(layer)

    def _update_camera_depth(self):
        """When the layer extents change, update the camera depth.

        The camera depth is the difference between the near clipping plane
        and the far clipping plane in a scene. If they are set too high
        relative to the actual depth of a scene, precision issues can arise
        in the depth of objects in the scene, with objects at the back
        seeming to pop to the front.

        See: https://github.com/napari/napari/issues/2138
        """
        if self.viewer.dims.ndisplay == 2:
            # don't bother updating 3D camera if we're not using it
            return
        # otherwise, set depth to diameter of displayed dimensions
        extent = self.viewer.layers.extent
        # we add a step because the difference is *right* at the point
        # coordinates, not accounting for voxel size:
        # >>> viewer.add_image(np.random.random((2, 3, 4, 5)))
        # >>> viewer.layers.extent
        # Extent(
        #     data=None,
        #     world=array([[0., 0., 0., 0.],
        #                  [1., 2., 3., 4.]]),
        #     step=array([1., 1., 1., 1.]),
        # )
        extent_all = extent.world[1] - extent.world[0] + extent.step
        extent_displayed = extent_all[list(self.viewer.dims.displayed)]
        diameter = np.linalg.norm(extent_displayed)
        # Use 128x the diameter to avoid aggressive near- and far-plane
        # clipping in perspective projection, while still preserving enough
        # bit depth in the depth buffer to avoid artifacts. See discussion at:
        # https://github.com/napari/napari/pull/7529#issuecomment-2594203871
        self.canvas.camera._3D_camera.depth_value = 128 * diameter

    def _add_layer(self, layer):
        """When a layer is added, set its parent and order.

        Parameters
        ----------
        layer : napari.layers.Layer
            Layer to be added.
        """
        vispy_layer = create_vispy_layer(layer)

        # QtPoll is experimental.
        if self._qt_poll is not None:
            # QtPoll will call VipyBaseImage._on_poll() when the camera
            # moves or the timer goes off.
            self._qt_poll.events.poll.connect(vispy_layer._on_poll)

            # In the other direction, some visuals need to tell QtPoll to
            # start polling. When they receive new data they need to be
            # polled to load it, even if the camera is not moving.
            if vispy_layer.events is not None:
                vispy_layer.events.loaded.connect(self._qt_poll.wake_up)

        self.canvas.add_layer_visual_mapping(layer, vispy_layer)

    def _remove_invalid_chars(self, selected_layer_name):
        """Removes invalid characters from selected layer name to suggest a filename.

        Parameters
        ----------
        selected_layer_name : str
            The selected napari layer name.

        Returns
        -------
        suggested_name : str
            Suggested name from input selected layer name, without invalid characters.
        """
        unprintable_ascii_chars = (
            '\x00',
            '\x01',
            '\x02',
            '\x03',
            '\x04',
            '\x05',
            '\x06',
            '\x07',
            '\x08',
            '\x0e',
            '\x0f',
            '\x10',
            '\x11',
            '\x12',
            '\x13',
            '\x14',
            '\x15',
            '\x16',
            '\x17',
            '\x18',
            '\x19',
            '\x1a',
            '\x1b',
            '\x1c',
            '\x1d',
            '\x1e',
            '\x1f',
            '\x7f',
        )
        invalid_characters = (
            ''.join(unprintable_ascii_chars)
            + '/'
            + '\\'  # invalid Windows filename character
            + ':*?"<>|\t\n\r\x0b\x0c'  # invalid Windows path characters
        )
        translation_table = dict.fromkeys(map(ord, invalid_characters), None)
        # Remove invalid characters
        suggested_name = selected_layer_name.translate(translation_table)
        return suggested_name

    def _save_layers_dialog(self, selected=False):
        """Save layers (all or selected) to disk, using ``LayerList.save()``.

        Parameters
        ----------
        selected : bool
            If True, only layers that are selected in the viewer will be saved.
            By default, all layers are saved.
        """
        msg = ''
        if not len(self.viewer.layers):
            msg = trans._('There are no layers in the viewer to save')
        elif len(self.viewer.layers) == 1:
            self.viewer.layers.selection.active = self.viewer.layers[0]
        elif selected and not len(self.viewer.layers.selection):
            msg = trans._(
                'Please select one or more layers to save,'
                '\nor use "Save all layers..."'
            )
        if msg:
            raise OSError(trans._('Nothing to save'))

        # prepare list of extensions for drop down menu.
        ext_str, writers = _extension_string_for_layers(
            list(self.viewer.layers.selection)
            if selected
            else self.viewer.layers
        )

        msg = trans._('selected') if selected else trans._('all')
        dlg = QFileDialog()
        hist = get_save_history()
        dlg.setHistory(hist)
        # get the layer's name to use for a default name if only one layer is selected
        selected_layer_name = ''
        if self.viewer.layers.selection.active is not None:
            selected_layer_name = self.viewer.layers.selection.active.name
            selected_layer_name = self._remove_invalid_chars(
                selected_layer_name
            )
        filename, selected_filter = dlg.getSaveFileName(
            self,  # parent
            trans._('Save {msg} layers', msg=msg),  # caption
            # home dir by default if selected all, home dir and file name if only 1 layer
            str(
                Path(hist[0]) / selected_layer_name
            ),  # directory in PyQt, dir in PySide
            filter=ext_str,
            options=(
                QFileDialog.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )
        logging.debug(
            trans._(
                'QFileDialog - filename: {filename} '
                'selected_filter: {selected_filter}',
                filename=filename or None,
                selected_filter=selected_filter or None,
            )
        )

        if filename:
            writer = _npe2_decode_selected_filter(
                ext_str, selected_filter, writers
            )
            with warnings.catch_warnings(record=True) as wa:
                saved = self.viewer.layers.save(
                    filename, selected=selected, _writer=writer
                )
                logging.debug('Saved %s', saved)
                error_messages = '\n'.join(str(x.message.args[0]) for x in wa)

            if not saved:
                raise OSError(
                    trans._(
                        'File {filename} save failed.\n{error_messages}',
                        deferred=True,
                        filename=filename,
                        error_messages=error_messages,
                    )
                )

            update_save_history(saved[0])

    def _update_welcome_screen(self):
        """Update welcome screen display based on layer count."""
        if self._show_welcome_screen:
            self._welcome_widget.set_welcome_visible(not self.viewer.layers)

    def _screenshot(self, flash=True):
        """Capture a screenshot of the Vispy canvas.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        """
        img = self.canvas.screenshot()
        if flash:
            from napari._qt.utils import add_flash_animation

            # Here we are actually applying the effect to the `_welcome_widget`
            # and not # the `native` widget because it does not work on the
            # `native` widget. It's probably because the widget is in a stack
            # with the `QtWelcomeWidget`.
            add_flash_animation(self._welcome_widget)
        return img

    def screenshot(self, path=None, flash=True) -> np.ndarray:
        """Take currently displayed screen and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """

        img = QImg2array(self._screenshot(flash))
        if path is not None:
            imsave(path, img)  # scikit-image imsave method
        return img

    def clipboard(self, flash=True):
        """Take a screenshot of the currently displayed screen and copy the
        image to the clipboard.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        """
        cb = QGuiApplication.clipboard()
        cb.setImage(self._screenshot(flash))

    def _screenshot_dialog(self):
        """Save screenshot of current display, default .png"""
        hist = get_save_history()
        dial = ScreenshotDialog(self.screenshot, self, hist[0], hist)
        if dial.exec_():
            update_save_history(dial.selectedFiles()[0])

    def _open_file_dialog_uni(self, caption: str) -> list[str]:
        """
        Open dialog to get list of files from user
        """
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)

        open_kwargs = {
            'parent': self,
            'caption': caption,
        }
        if 'pyside' in QFileDialog.__module__.lower():
            # PySide6
            open_kwargs['dir'] = hist[0]
        else:
            open_kwargs['directory'] = hist[0]

        if in_ipython():
            open_kwargs['options'] = QFileDialog.DontUseNativeDialog

        return dlg.getOpenFileNames(**open_kwargs)[0]

    def _open_files_dialog(self, choose_plugin=False, stack=False):
        """Add files from the menubar."""
        filenames = self._open_file_dialog_uni(trans._('Select file(s)...'))

        if filenames:
            self._qt_open(filenames, choose_plugin=choose_plugin, stack=stack)
            update_open_history(filenames[0])

    def _open_files_dialog_as_stack_dialog(self, choose_plugin=False):
        """Add files as a stack, from the menubar."""
        return self._open_files_dialog(choose_plugin=choose_plugin, stack=True)

    def _open_folder_dialog(self, choose_plugin=False):
        """Add a folder of files from the menubar."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)

        folder = dlg.getExistingDirectory(
            self,
            trans._('Select folder...'),
            hist[0],  # home dir by default
            (
                QFileDialog.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )

        if folder not in {'', None}:
            self._qt_open([folder], stack=False, choose_plugin=choose_plugin)
            update_open_history(folder)

    def _qt_open(
        self,
        filenames: list[str],
        stack: Union[bool, list[list[str]]],
        choose_plugin: bool = False,
        plugin: Optional[str] = None,
        layer_type: Optional[str] = None,
        **kwargs,
    ):
        """Open files, potentially popping reader dialog for plugin selection.

        Call ViewerModel.open and catch errors that could
        be fixed by user making a plugin choice.

        Parameters
        ----------
        filenames : List[str]
            paths to open
        choose_plugin : bool
            True if user wants to explicitly choose the plugin else False
        stack : bool or list[list[str]]
            whether to stack files or not. Can also be a list containing
            files to stack.
        plugin : str
            plugin to use for reading
        layer_type : str
            layer type for opened layers
        """
        if choose_plugin:
            handle_gui_reading(
                filenames, self, stack, plugin_override=choose_plugin, **kwargs
            )
            return

        try:
            self.viewer.open(
                filenames,
                stack=stack,
                plugin=plugin,
                layer_type=layer_type,
                **kwargs,
            )
        except ReaderPluginError as e:
            handle_gui_reading(
                filenames,
                self,
                stack,
                e.reader_plugin,
                e,
                layer_type=layer_type,
                **kwargs,
            )
        except MultipleReaderError:
            handle_gui_reading(filenames, self, stack, **kwargs)

    def toggle_console_visibility(self, event=None):
        """Toggle console visible and not visible.

        Imports the console the first time it is requested.
        """
        if in_ipython() or in_jupyter():
            return

        # force instantiation of console if not already instantiated
        _ = self.console

        viz = not self.dockConsole.isVisible()
        # modulate visibility at the dock widget level as console is dockable
        self.dockConsole.setVisible(viz)
        if self.dockConsole.isFloating():
            self.dockConsole.setFloating(True)

        if viz:
            self.dockConsole.raise_()
            self.dockConsole.setFocus()

        self.viewerButtons.consoleButton.setProperty(
            'expanded', self.dockConsole.isVisible()
        )
        self.viewerButtons.consoleButton.style().unpolish(
            self.viewerButtons.consoleButton
        )
        self.viewerButtons.consoleButton.style().polish(
            self.viewerButtons.consoleButton
        )

    def set_welcome_visible(self, visible):
        """Show welcome screen widget."""
        self._show_welcome_screen = visible
        self._welcome_widget.set_welcome_visible(visible)

    def keyPressEvent(self, event):
        """Called whenever a key is pressed.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self.canvas._scene_canvas._backend._keyEvent(
            self.canvas._scene_canvas.events.key_press, event
        )
        event.accept()

    def keyReleaseEvent(self, event):
        """Called whenever a key is released.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self.canvas._scene_canvas._backend._keyEvent(
            self.canvas._scene_canvas.events.key_release, event
        )
        event.accept()

    def dragEnterEvent(self, event):
        """Ignore event if not dragging & dropping a file or URL to open.

        Using event.ignore() here allows the event to pass through the
        parent widget to its child widget, otherwise the parent widget
        would catch the event and not pass it on to the child widget.

        Parameters
        ----------
        event : qtpy.QtCore.QDragEvent
            Event from the Qt context.
        """
        if event.mimeData().hasUrls():
            self._set_drag_status()
            event.accept()
        else:
            event.ignore()

    def _set_drag_status(self):
        """Set dedicated status message when dragging files into viewer"""
        self.viewer.status = trans._(
            'Hold <Alt> key to open plugin selection. Hold <Shift> to open files as stack.'
        )

    def _image_from_clipboard(self):
        """Insert image from clipboard as a new layer if clipboard contains an image or link."""
        cb = QGuiApplication.clipboard()
        if cb.mimeData().hasImage():
            image = cb.image()
            if image.isNull():
                return
            arr = QImg2array(image)
            self.viewer.add_image(arr)
            return
        if cb.mimeData().hasUrls():
            show_info('No image in clipboard, trying to open link instead.')
            self._open_from_list_of_urls_data(
                cb.mimeData().urls(), stack=False, choose_plugin=False
            )
            return
        if cb.mimeData().hasText():
            show_info(
                'No image in clipboard, trying to parse text in clipboard as a link.'
            )
            url_list = []
            for line in cb.mimeData().text().split('\n'):
                url = QUrl(line.strip())
                if url.isEmpty():
                    continue
                if url.scheme() == '':
                    url.setScheme('file')
                if url.isLocalFile() and not Path(url.toLocalFile()).exists():
                    break
                url_list.append(url)
            else:
                self._open_from_list_of_urls_data(
                    url_list, stack=False, choose_plugin=False
                )
                return
        show_info('No image or link in clipboard.')

    def dropEvent(self, event):
        """Add local files and web URLS with drag and drop.

        For each file, attempt to open with existing associated reader
        (if available). If no reader is associated or opening fails,
        and more than one reader is available, open dialog and ask
        user to choose among available readers. User can choose to persist
        this choice.

        Parameters
        ----------
        event : qtpy.QtCore.QDropEvent
            Event from the Qt context.
        """
        shift_down = (
            QGuiApplication.keyboardModifiers()
            & Qt.KeyboardModifier.ShiftModifier
        )
        alt_down = (
            QGuiApplication.keyboardModifiers()
            & Qt.KeyboardModifier.AltModifier
        )
        self._open_from_list_of_urls_data(
            event.mimeData().urls(),
            stack=bool(shift_down),
            choose_plugin=bool(alt_down),
        )

    def _open_from_list_of_urls_data(
        self, urls_list: list[QUrl], stack: bool, choose_plugin: bool
    ):
        filenames = []
        for url in urls_list:
            if url.isLocalFile():
                # directories get a trailing "/", Path conversion removes it
                filenames.append(str(Path(url.toLocalFile())))
            else:
                filenames.append(url.toString())

        self._qt_open(
            filenames,
            stack=stack,
            choose_plugin=choose_plugin,
        )

    def closeEvent(self, event):
        """Cleanup and close.

        Parameters
        ----------
        event : qtpy.QtCore.QCloseEvent
            Event from the Qt context.
        """
        if self._layers is not None:
            # do not create layerlist if it does not exist yet.
            self.layers.close()

        # if the viewer.QtDims object is playing an axis, we need to terminate
        # the AnimationThread before close, otherwise it will cause a segFault
        # or Abort trap. (calling stop() when no animation is occurring is also
        # not a problem)
        self.dims.stop()
        self.canvas.delete()
        if self._console is not None:
            self.console.close()
        self.dockConsole.deleteLater()
        event.accept()


if TYPE_CHECKING:
    from napari._qt.experimental.qt_poll import QtPoll
    from napari.components.experimental.remote import RemoteManager


def _create_qt_poll(parent: QObject, camera: Camera) -> Optional[QtPoll]:
    """Create and return a QtPoll instance, if needed.

    Create a QtPoll instance for the monitor.

    Monitor needs QtPoll to poll for incoming messages. This might be
    temporary until we can process incoming messages with a dedicated
    thread.

    Parameters
    ----------
    parent : QObject
        Parent Qt object.
    camera : Camera
        Camera that the QtPoll object will listen to.

    Returns
    -------
    Optional[QtPoll]
        The new QtPoll instance, if we need one.
    """
    if not config.monitor:
        return None

    from napari._qt.experimental.qt_poll import QtPoll

    qt_poll = QtPoll(parent)
    camera.events.connect(qt_poll.on_camera)
    return qt_poll


def _create_remote_manager(
    layers: LayerList, qt_poll
) -> Optional[RemoteManager]:
    """Create and return a RemoteManager instance, if we need one.

    Parameters
    ----------
    layers : LayersList
        The viewer's layers.
    qt_poll : QtPoll
        The viewer's QtPoll instance.
    """
    if not config.monitor:
        return None  # Not using the monitor at all

    from napari.components.experimental.monitor import monitor
    from napari.components.experimental.remote import RemoteManager

    # Start the monitor so we can access its events. The monitor has no
    # dependencies to napari except to utils.Event.
    started = monitor.start()

    if not started:
        return None  # Probably not >= Python 3.9, so no manager is needed.

    # Create the remote manager and have monitor call its process_command()
    # method to execute commands from clients.
    manager = RemoteManager(layers)

    # RemoteManager will process incoming command from the monitor.
    monitor.run_command_event.connect(manager.process_command)

    # QtPoll should pool the RemoteManager and the Monitor.
    qt_poll.events.poll.connect(manager.on_poll)
    qt_poll.events.poll.connect(monitor.on_poll)

    return manager


def _in_napari(n: int, frame: FrameType):
    """
    Determines whether we are in napari by looking at:
        1) the frames modules names:
        2) the min_depth
    """
    if n < 2:
        return True
    # in-n-out is used in napari for dependency injection.
    for pref in {'napari.', 'in_n_out.'}:
        if frame.f_globals.get('__name__', '').startswith(pref):
            return True
    return False
