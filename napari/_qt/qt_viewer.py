from __future__ import annotations

import logging
import traceback
import typing
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Type, Union
from weakref import WeakSet

from qtpy.QtCore import QCoreApplication, QObject, Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QFileDialog, QSplitter, QVBoxLayout, QWidget
from superqt import ensure_main_thread

from napari._qt.containers import QtLayerTreeView
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
from napari.utils.translations import trans
from napari_builtins.io import imsave_extensions

from napari._vispy import VispyCanvas, create_vispy_layer  # isort:skip

if TYPE_CHECKING:
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
        ext_str.split(";;"),
        writers,
    ):
        if entry.startswith(selected_filter):
            return writer
    return None


def _extension_string_for_layers(
    layers: Sequence[Layer],
) -> Tuple[str, List[WriterContribution]]:
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

            ext_list = [f"*{val}" for val in ext]
            ext_str = ';;'.join(ext_list)

            ext_str = trans._(
                "All Files (*);; Image file types:;;{ext_str}",
                ext_str=ext_str,
            )

        elif selected_layer._type_string == 'points':
            ext_str = trans._("All Files (*);; *.csv;;")

        else:
            # layer other than image or points
            ext_str = trans._("All Files (*);;")

    else:
        # multiple layers.
        ext_str = trans._("All Files (*);;")
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
        A QtPoll object required for octree or monitor.
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
        canvas_class: Type[VispyCanvas] = VispyCanvas,
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
        self.viewer.layers.events.inserted.connect(self._update_welcome_screen)
        self.viewer.layers.events.removed.connect(self._update_welcome_screen)
        self.viewer.layers.selection.events.active.connect(
            self._on_active_change
        )

        self.viewer.layers.events.inserted.connect(self._on_add_layer_change)

        self.setAcceptDrops(True)

        # Create the experimental QtPool for octree and/or monitor.
        self._qt_poll = _create_qt_poll(self, self.viewer.camera)

        # Create the experimental RemoteManager for the monitor.
        self._remote_manager = _create_remote_manager(
            self.viewer.layers, self._qt_poll
        )

        # moved from the old layerlist... still feels misplaced.
        # can you help me move this elsewhere?
        if config.async_loading:
            from napari._qt.experimental.qt_chunk_receiver import (
                QtChunkReceiver,
            )

            # The QtChunkReceiver object allows the ChunkLoader to pass newly
            # loaded chunks to the layers that requested them.
            self.chunk_receiver = QtChunkReceiver(self.layers)
        else:
            self.chunk_receiver = None

        # bind shortcuts stored in settings last.
        self._bind_shortcuts()

        settings = get_settings()
        self._update_dask_cache_settings(settings.application.dask)

        settings.application.events.dask.connect(
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
                "Access to QtViewer.view is deprecated since 0.5.0 and will be removed in the napari 0.6.0. Change to QtViewer.canvas.view instead."
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
                "Access to QtViewer.camera will become deprecated in the 0.6.0. Change to QtViewer.canvas.camera instead."
            ),
            FutureWarning,
            stacklevel=2,
        )
        return self.canvas.camera

    @staticmethod
    def _update_dask_cache_settings(
        dask_setting: Union[DaskSettings, Event] = None
    ):
        """Update dask cache to match settings."""
        if not dask_setting:
            return
        if not isinstance(dask_setting, DaskSettings):
            dask_setting = dask_setting.value

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
    def layers(self) -> QtLayerTreeView:
        if self._layers is None:
            self._layers = QtLayerTreeView(self.viewer.layers)
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
        self.viewer.status = ""
        self.viewer.mouse_over_canvas = False

    def _enter_canvas(self):
        """enable status on canvas enter"""
        self.viewer.status = "Ready"
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
        if perf.USE_PERFMON:
            return QtViewerDockWidget(
                self,
                QtPerformance(),
                name=trans._('performance'),
                area='bottom',
            )
        return None

    @property
    def console(self):
        """QtConsole: iPython console terminal integrated into the napari GUI."""
        if self._console is None:
            try:
                from napari_console import QtConsole

                import napari

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self.console = QtConsole(self.viewer)
                    self.console.push(
                        {'napari': napari, 'action_manager': action_manager}
                    )
            except ModuleNotFoundError:
                warnings.warn(
                    trans._(
                        'napari-console not found. It can be installed with'
                        ' "pip install napari_console"'
                    ),
                    stacklevel=1,
                )
                self._console = None
            except ImportError:
                traceback.print_exc()
                warnings.warn(
                    trans._(
                        'error importing napari-console. See console for full error.'
                    ),
                    stacklevel=1,
                )
                self._console = None
        return self._console

    @console.setter
    def console(self, console):
        self._console = console
        if console is not None:
            self.dockConsole.setWidget(console)
            console.setParent(self.dockConsole)

    @ensure_main_thread
    def _on_slice_ready(self, event):
        responses = event.value
        logging.debug('QtViewer._on_slice_ready: %s', responses)
        for layer, response in responses.items():
            # Update the layer slice state to temporarily support behavior
            # that depends on it.
            layer._update_slice_response(response)
            # The rest of `Layer.refresh` after `set_view_slice`, where `set_data`
            # notifies the corresponding vispy layer of the new slice.
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

    def _add_layer(self, layer):
        """When a layer is added, set its parent and order.

        Parameters
        ----------
        layer : napari.layers.Layer
            Layer to be added.
        """

        def add_children(layer):
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

            if layer.is_group():
                for child in layer:
                    add_children(child)

        add_children(layer)

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
            msg = trans._("There are no layers in the viewer to save")
        elif selected and not len(self.viewer.layers.selection):
            msg = trans._(
                'Please select one or more layers to save,'
                '\nor use "Save all layers..."'
            )
        if msg:
            raise OSError(trans._("Nothing to save"))

        # prepare list of extensions for drop down menu.
        ext_str, writers = _extension_string_for_layers(
            list(self.viewer.layers.selection)
            if selected
            else self.viewer.layers
        )

        msg = trans._("selected") if selected else trans._("all")
        dlg = QFileDialog()
        hist = get_save_history()
        dlg.setHistory(hist)

        filename, selected_filter = dlg.getSaveFileName(
            parent=self,
            caption=trans._('Save {msg} layers', msg=msg),
            directory=hist[0],  # home dir by default,
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
                error_messages = "\n".join(str(x.message.args[0]) for x in wa)

            if not saved:
                raise OSError(
                    trans._(
                        "File {filename} save failed.\n{error_messages}",
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
        # CAN REMOVE THIS AFTER DEPRECATION IS DONE, see self.screenshot.
        img = self.canvas.screenshot()
        if flash:
            from napari._qt.utils import add_flash_animation

            # Here we are actually applying the effect to the `_welcome_widget`
            # and not # the `native` widget because it does not work on the
            # `native` widget. It's probably because the widget is in a stack
            # with the `QtWelcomeWidget`.
            add_flash_animation(self._welcome_widget)
        return img

    def screenshot(self, path=None, flash=True):
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

    def _open_file_dialog_uni(self, caption: str) -> typing.List[str]:
        """
        Open dialog to get list of files from user
        """
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)

        open_kwargs = {
            "parent": self,
            "caption": caption,
        }
        if "pyside" in QFileDialog.__module__.lower():
            # PySide6
            open_kwargs["dir"] = hist[0]
        else:
            open_kwargs["directory"] = hist[0]

        if in_ipython():
            open_kwargs["options"] = QFileDialog.DontUseNativeDialog

        return dlg.getOpenFileNames(**open_kwargs)[0]

    def _open_files_dialog(self, choose_plugin=False):
        """Add files from the menubar."""
        filenames = self._open_file_dialog_uni(trans._('Select file(s)...'))

        if (filenames != []) and (filenames is not None):
            for filename in filenames:
                self._qt_open(
                    [filename], stack=False, choose_plugin=choose_plugin
                )
            update_open_history(filenames[0])

    def _open_files_dialog_as_stack_dialog(self, choose_plugin=False):
        """Add files as a stack, from the menubar."""

        filenames = self._open_file_dialog_uni(trans._('Select files...'))

        if (filenames != []) and (filenames is not None):
            self._qt_open(filenames, stack=True, choose_plugin=choose_plugin)
            update_open_history(filenames[0])

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
        filenames: List[str],
        stack: Union[bool, List[List[str]]],
        choose_plugin: bool = False,
        plugin: str = None,
        layer_type: str = None,
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

    def _toggle_chunk_outlines(self):
        """Toggle whether we are drawing outlines around the chunks."""
        from napari.layers.image.experimental.octree_image import (
            _OctreeImageBase,
        )

        for layer in self.viewer.layers:
            if isinstance(layer, _OctreeImageBase):
                layer.display.show_grid = not layer.display.show_grid

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
        filenames = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                # directories get a trailing "/", Path conversion removes it
                filenames.append(str(Path(url.toLocalFile())))
            else:
                filenames.append(url.toString())

        self._qt_open(
            filenames,
            stack=bool(shift_down),
            choose_plugin=bool(alt_down),
        )

    def closeEvent(self, event):
        """Cleanup and close.

        Parameters
        ----------
        event : qtpy.QtCore.QCloseEvent
            Event from the Qt context.
        """
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

    Create a QtPoll instance for octree or monitor.

    Octree needs QtPoll so VispyTiledImageLayer can finish in-progress
    loads even if the camera is not moving. Once loading is finish it will
    tell QtPoll it no longer needs to be polled.

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
    if not config.async_octree and not config.monitor:
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
