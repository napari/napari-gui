"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""

import contextlib
import inspect
import os
import sys
import time
import warnings
from collections.abc import MutableMapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Optional,
    Union,
    cast,
)
from weakref import WeakValueDictionary

import numpy as np
from qtpy.QtCore import (
    QEvent,
    QEventLoop,
    QPoint,
    QProcess,
    QRect,
    QSize,
    Qt,
    Slot,
)
from qtpy.QtGui import QHideEvent, QIcon, QShowEvent
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QShortcut,
    QToolTip,
    QWidget,
)

from napari._app_model.constants import MenuId
from napari._app_model.context import create_context, get_context
from napari._qt._qapp_model import build_qmodel_menu
from napari._qt._qapp_model.qactions import add_dummy_actions, init_qactions
from napari._qt._qapp_model.qactions._debug import _is_set_trace_active
from napari._qt._qplugins import (
    _rebuild_npe1_plugins_menu,
    _rebuild_npe1_samples_menu,
)
from napari._qt.dialogs.confirm_close_dialog import ConfirmCloseDialog
from napari._qt.dialogs.preferences_dialog import PreferencesDialog
from napari._qt.dialogs.qt_activity_dialog import QtActivityDialog
from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari._qt.qt_event_loop import (
    NAPARI_ICON_PATH,
    get_qapp,
    quit_app as quit_app_,
)
from napari._qt.qt_resources import get_stylesheet
from napari._qt.qt_viewer import QtViewer
from napari._qt.threads.status_checker import StatusChecker
from napari._qt.utils import QImg2array, qbytearray_to_str, str_to_qbytearray
from napari._qt.widgets.qt_viewer_dock_widget import (
    _SHORTCUT_DEPRECATION_STRING,
    QtViewerDockWidget,
)
from napari._qt.widgets.qt_viewer_status_bar import ViewerStatusBar
from napari.plugins import (
    menu_item_template as plugin_menu_item_template,
    plugin_manager,
)
from napari.plugins._npe2 import index_npe1_adapters
from napari.settings import get_settings
from napari.utils import perf
from napari.utils._proxies import PublicOnlyProxy
from napari.utils.events import Event
from napari.utils.io import imsave
from napari.utils.misc import (
    in_ipython,
    in_jupyter,
    in_python_repl,
    running_as_constructor_app,
)
from napari.utils.notifications import Notification
from napari.utils.theme import _themes, get_system_theme
from napari.utils.translations import trans

_sentinel = object()

if TYPE_CHECKING:
    from magicgui.widgets import Widget
    from qtpy.QtGui import QImage

    from napari.viewer import Viewer


MenuStr = Literal[
    'file_menu',
    'view_menu',
    'layers_menu',
    'plugins_menu',
    'window_menu',
    'help_menu',
]


class _QtMainWindow(QMainWindow):
    # This was added so that someone can patch
    # `napari._qt.qt_main_window._QtMainWindow._window_icon`
    # to their desired window icon
    _window_icon = NAPARI_ICON_PATH

    # To track window instances and facilitate getting the "active" viewer...
    # We use this instead of QApplication.activeWindow for compatibility with
    # IPython usage. When you activate IPython, it will appear that there are
    # *no* active windows, so we want to track the most recently active windows
    _instances: ClassVar[list['_QtMainWindow']] = []

    # `window` is passed through on construction, so it's available to a window
    # provider for dependency injection
    # See https://github.com/napari/napari/pull/4826
    def __init__(
        self, viewer: 'Viewer', window: 'Window', parent=None
    ) -> None:
        super().__init__(parent)
        self._ev = None
        self._window = window
        self._qt_viewer = QtViewer(viewer, show_welcome_screen=True)
        self._quit_app = False

        self.setWindowIcon(QIcon(self._window_icon))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        center = QWidget(self)
        center.setLayout(QHBoxLayout())
        center.layout().addWidget(self._qt_viewer)
        center.layout().setContentsMargins(4, 0, 4, 0)
        self.setCentralWidget(center)

        self.setWindowTitle(self._qt_viewer.viewer.title)

        self._maximized_flag = False
        self._normal_geometry = QRect()
        self._window_size = None
        self._window_pos = None
        self._old_size = None
        self._positions = []
        self._toggle_menubar_visibility = False

        self._is_close_dialog = {False: True, True: True}
        # this ia sa workaround for #5335 issue. The dict is used to not
        # collide shortcuts for close and close all windows

        act_dlg = QtActivityDialog(self._qt_viewer._welcome_widget)
        self._qt_viewer._welcome_widget.resized.connect(
            act_dlg.move_to_bottom_right
        )
        act_dlg.hide()
        self._activity_dialog = act_dlg

        self.setStatusBar(ViewerStatusBar(self))

        # Prevent QLineEdit based widgets to keep focus even when clicks are
        # done outside the widget. See #1571
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Ideally this would be in `NapariApplication` but that is outside of Qt
        self._viewer_context = create_context(self)
        self._viewer_context['is_set_trace_active'] = _is_set_trace_active

        settings = get_settings()

        # TODO:
        # settings.plugins.defaults.call_order = plugin_manager.call_order()

        # set the values in plugins to match the ones saved in settings
        if settings.plugins.call_order is not None:
            plugin_manager.set_call_order(settings.plugins.call_order)

        _QtMainWindow._instances.append(self)

        # since we initialize canvas before the window,
        # we need to manually connect them again.
        handle = self.windowHandle()
        if handle is not None:
            handle.screenChanged.connect(self._qt_viewer.canvas.screen_changed)

        # this is the line that initializes any Qt-based app-model Actions that
        # were defined somewhere in the `_qt` module and imported in init_qactions
        init_qactions()

        with contextlib.suppress(IndexError):
            viewer.cursor.events.position.disconnect(
                viewer.update_status_from_cursor
            )

        self.status_thread = StatusChecker(viewer, parent=self)
        self.status_thread.status_and_tooltip_changed.connect(
            self.set_status_and_tooltip
        )
        viewer.cursor.events.position.connect(
            self.status_thread.trigger_status_update
        )
        settings.appearance.events.update_status_based_on_layer.connect(
            self._toggle_status_thread
        )

    def _toggle_status_thread(self, event: Event):
        if event.value:
            self.status_thread.start()
        else:
            self.status_thread.terminate()

    def showEvent(self, event: QShowEvent):
        """Override to handle window state changes."""
        settings = get_settings()
        # if event loop is not running, we don't want to start the thread
        # If event loop is running, the loopLevel will be above 0
        if (
            settings.appearance.update_status_based_on_layer
            and QApplication.instance().thread().loopLevel()
        ):
            self.status_thread.start()
        super().showEvent(event)

    def enterEvent(self, a0):
        # as we call show in Viewer constructor, we need to start the thread
        # when the mouse enters the window
        # as first call of showEvent is before the event loop is running
        if (
            get_settings().appearance.update_status_based_on_layer
            and not self.status_thread.isRunning()
        ):
            self.status_thread.start()
        super().enterEvent(a0)

    def hideEvent(self, event: QHideEvent):
        self.status_thread.terminate()
        super().hideEvent(event)

    def set_status_and_tooltip(
        self, status_and_tooltip: tuple[str | dict, str] | None
    ):
        if status_and_tooltip is None:
            return
        self._qt_viewer.viewer.status = status_and_tooltip[0]
        self._qt_viewer.viewer.tooltip.text = status_and_tooltip[1]
        if (
            active := self._qt_viewer.viewer.layers.selection.active
        ) is not None:
            self._qt_viewer.viewer.help = active.help

    def statusBar(self) -> 'ViewerStatusBar':
        return super().statusBar()

    @classmethod
    def current(cls) -> Optional['_QtMainWindow']:
        return cls._instances[-1] if cls._instances else None

    @classmethod
    def current_viewer(cls):
        window = cls.current()
        return window._qt_viewer.viewer if window else None

    def event(self, e: QEvent) -> bool:
        if (
            e.type() == QEvent.Type.ToolTip
            and self._qt_viewer.viewer.tooltip.visible
        ):
            # globalPos is for Qt5 e.globalPosition().toPoint() is for QT6
            # https://doc-snapshots.qt.io/qt6-dev/qmouseevent-obsolete.html#globalPos
            pnt = (
                e.globalPosition().toPoint()
                if hasattr(e, 'globalPosition')
                else e.globalPos()
            )
            QToolTip.showText(pnt, self._qt_viewer.viewer.tooltip.text, self)
        if e.type() in {QEvent.Type.WindowActivate, QEvent.Type.ZOrderChange}:
            # upon activation or raise_, put window at the end of _instances
            with contextlib.suppress(ValueError):
                inst = _QtMainWindow._instances
                inst.append(inst.pop(inst.index(self)))

        res = super().event(e)

        if e.type() == QEvent.Type.Close and e.isAccepted():
            # when we close the MainWindow, remove it from the instance list
            with contextlib.suppress(ValueError):
                _QtMainWindow._instances.remove(self)

        return res

    def showFullScreen(self):
        super().showFullScreen()
        # Handle OpenGL based windows fullscreen issue on Windows.
        # For more info see:
        #  * https://doc.qt.io/qt-6/windows-issues.html#fullscreen-opengl-based-windows
        #  * https://bugreports.qt.io/browse/QTBUG-41309
        #  * https://bugreports.qt.io/browse/QTBUG-104511
        if os.name != 'nt':
            return
        import win32con
        import win32gui

        if self.windowHandle():
            handle = int(self.windowHandle().winId())
            win32gui.SetWindowLong(
                handle,
                win32con.GWL_STYLE,
                win32gui.GetWindowLong(handle, win32con.GWL_STYLE)
                | win32con.WS_BORDER,
            )

    def eventFilter(self, source, event):
        # Handle showing hidden menubar on mouse move event.
        # We do not hide menubar when a menu is being shown or
        # we are not in menubar toggled state
        if (
            QApplication.activePopupWidget() is None
            and hasattr(self, '_toggle_menubar_visibility')
            and self._toggle_menubar_visibility
        ):
            if event.type() == QEvent.Type.MouseMove:
                if self.menuBar().isHidden():
                    rect = self.geometry()
                    # set mouse-sensitive zone to trigger showing the menubar
                    rect.setHeight(25)
                    if rect.contains(event.globalPos()):
                        self.menuBar().show()
                else:
                    rect = QRect(
                        self.menuBar().mapToGlobal(QPoint(0, 0)),
                        self.menuBar().size(),
                    )
                    if not rect.contains(event.globalPos()):
                        self.menuBar().hide()
            elif event.type() == QEvent.Type.Leave and source is self:
                self.menuBar().hide()
        return super().eventFilter(source, event)

    def _load_window_settings(self):
        """
        Load window layout settings from configuration.
        """
        settings = get_settings()
        window_position = settings.application.window_position

        # It's necessary to verify if the window/position value is valid with
        # the current screen.
        if not window_position:
            window_position = (self.x(), self.y())
        else:
            origin_x, origin_y = window_position
            screen = QApplication.screenAt(QPoint(origin_x, origin_y))
            screen_geo = screen.geometry() if screen else None
            if not screen_geo:
                window_position = (self.x(), self.y())

        return (
            settings.application.window_state,
            settings.application.window_size,
            window_position,
            settings.application.window_maximized,
            settings.application.window_fullscreen,
        )

    def _get_window_settings(self):
        """Return current window settings.

        Symmetric to the 'set_window_settings' setter.
        """

        window_fullscreen = self.isFullScreen()
        if window_fullscreen:
            window_maximized = self._maximized_flag
        else:
            window_maximized = self.isMaximized()

        window_state = qbytearray_to_str(self.saveState())
        return (
            window_state,
            self._window_size or (self.width(), self.height()),
            self._window_pos or (self.x(), self.y()),
            window_maximized,
            window_fullscreen,
        )

    def _set_window_settings(
        self,
        window_state,
        window_size,
        window_position,
        window_maximized,
        window_fullscreen,
    ):
        """
        Set window settings.

        Symmetric to the 'get_window_settings' accessor.
        """
        self.setUpdatesEnabled(False)
        self.setWindowState(Qt.WindowState.WindowNoState)

        if window_position:
            window_position = QPoint(*window_position)
            self.move(window_position)

        if window_size:
            window_size = QSize(*window_size)
            self.resize(window_size)

        if window_state:
            self.restoreState(str_to_qbytearray(window_state))

        # Toggling the console visibility is disabled when it is not
        # available, so ensure that it is hidden.
        if in_ipython() or in_jupyter() or in_python_repl():
            self._qt_viewer.dockConsole.setVisible(False)

        if window_fullscreen:
            self._maximized_flag = window_maximized
            self.showFullScreen()
        elif window_maximized:
            self.setWindowState(Qt.WindowState.WindowMaximized)

        self.setUpdatesEnabled(True)

    def _save_current_window_settings(self):
        """Save the current geometry of the main window."""
        (
            window_state,
            window_size,
            window_position,
            window_maximized,
            window_fullscreen,
        ) = self._get_window_settings()

        settings = get_settings()
        if settings.application.save_window_geometry:
            settings.application.window_maximized = window_maximized
            settings.application.window_fullscreen = window_fullscreen
            settings.application.window_position = window_position
            settings.application.window_size = window_size
            settings.application.window_statusbar = (
                not self.statusBar().isHidden()
            )

        if settings.application.save_window_state:
            settings.application.window_state = window_state

    def close(self, quit_app=False, confirm_need=False):
        """Override to handle closing app or just the window."""
        if not quit_app and not self._qt_viewer.viewer.layers:
            return super().close()
        confirm_need_local = confirm_need and self._is_close_dialog[quit_app]
        self._is_close_dialog[quit_app] = False
        # here we save information that we could request confirmation on close
        # So fi function `close` is called again, we don't ask again but just close
        if (
            not confirm_need_local
            or not get_settings().application.confirm_close_window
            or ConfirmCloseDialog(self, quit_app).exec_() == QDialog.Accepted
        ):
            self._quit_app = quit_app
            self._is_close_dialog[quit_app] = True
            # here we inform that confirmation dialog is not open
            self._qt_viewer.dims.stop()
            return super().close()
        self._is_close_dialog[quit_app] = True
        return None
        # here we inform that confirmation dialog is not open

    def close_window(self):
        """Close active dialog or active window."""
        parent = QApplication.focusWidget()
        while parent is not None:
            if isinstance(parent, QMainWindow):
                self.close()
                break

            if isinstance(parent, QDialog):
                parent.close()
                break

            try:
                parent = parent.parent()
            except AttributeError:
                parent = getattr(parent, '_parent', None)

    def show(self, block=False):
        super().show()
        self._qt_viewer.setFocus()
        if block:
            self._ev = QEventLoop()
            self._ev.exec()

    def changeEvent(self, event):
        """Handle window state changes."""
        if event.type() == QEvent.Type.WindowStateChange:
            # TODO: handle maximization issue. When double clicking on the
            # title bar on Mac the resizeEvent is called an varying amount
            # of times which makes it hard to track the original size before
            # maximization.
            condition = (
                self.isMaximized() if os.name == 'nt' else self.isFullScreen()
            )
            if condition and self._old_size is not None:
                if self._positions and len(self._positions) > 1:
                    self._window_pos = self._positions[-2]

                self._window_size = (
                    self._old_size.width(),
                    self._old_size.height(),
                )
            else:
                self._old_size = None
                self._window_pos = None
                self._window_size = None
                self._positions = []

        super().changeEvent(event)

    def keyPressEvent(self, event):
        """Called whenever a key is pressed.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._qt_viewer.canvas._scene_canvas._backend._keyEvent(
            self._qt_viewer.canvas._scene_canvas.events.key_press, event
        )
        event.accept()

    def keyReleaseEvent(self, event):
        """Called whenever a key is released.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._qt_viewer.canvas._scene_canvas._backend._keyEvent(
            self._qt_viewer.canvas._scene_canvas.events.key_release, event
        )
        event.accept()

    def resizeEvent(self, event):
        """Override to handle original size before maximizing."""
        # the first resize event will have nonsense positions that we don't
        # want to store (and potentially restore)
        if event.oldSize().isValid():
            self._old_size = event.oldSize()
            self._positions.append((self.x(), self.y()))

            if self._positions and len(self._positions) >= 2:
                self._window_pos = self._positions[-2]
                self._positions = self._positions[-2:]

        super().resizeEvent(event)

    def closeEvent(self, event):
        """This method will be called when the main window is closing.

        Regardless of whether cmd Q, cmd W, or the close button is used...
        """
        if (
            event.spontaneous()
            and get_settings().application.confirm_close_window
            and self._qt_viewer.viewer.layers
            and ConfirmCloseDialog(self, False).exec_() != QDialog.Accepted
        ):
            event.ignore()
            return

        self.status_thread.terminate()
        self.status_thread.wait()

        if self._ev and self._ev.isRunning():
            self._ev.quit()

        # Close any floating dockwidgets
        for dock in self.findChildren(QtViewerDockWidget):
            if isinstance(dock, QWidget) and dock.isFloating():
                dock.setFloating(False)

        self._save_current_window_settings()

        # On some versions of Darwin, exiting while fullscreen seems to tickle
        # some bug deep in NSWindow.  This forces the fullscreen keybinding
        # test to complete its draw cycle, then pop back out of fullscreen.
        if self.isFullScreen():
            self.showNormal()
            for _ in range(5):
                time.sleep(0.1)
                QApplication.processEvents()

        self._qt_viewer.dims.stop()

        if self._quit_app:
            quit_app_()

        event.accept()

    def restart(self):
        """Restart the napari application in a detached process."""
        process = QProcess()
        process.setProgram(sys.executable)

        if not running_as_constructor_app():
            process.setArguments(sys.argv)

        process.startDetached()
        self.close(quit_app=True)

    def toggle_menubar_visibility(self):
        """
        Change menubar to be shown or to be hidden and shown on mouse movement.

        For the mouse movement functionality see the `eventFilter` implementation.
        """
        self._toggle_menubar_visibility = not self._toggle_menubar_visibility
        self.menuBar().setVisible(not self._toggle_menubar_visibility)
        return self._toggle_menubar_visibility

    @staticmethod
    @Slot(Notification)
    def show_notification(notification: Notification):
        """Show notification coming from a thread."""
        NapariQtNotification.show_notification(notification)


class Window:
    """Application window that contains the menu bar and viewer.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Contained viewer widget.

    Attributes
    ----------
    file_menu : qtpy.QtWidgets.QMenu
        File menu.
    help_menu : qtpy.QtWidgets.QMenu
        Help menu.
    main_menu : qtpy.QtWidgets.QMainWindow.menuBar
        Main menubar.
    view_menu : qtpy.QtWidgets.QMenu
        View menu.
    window_menu : qtpy.QtWidgets.QMenu
        Window menu.
    """

    def __init__(self, viewer: 'Viewer', *, show: bool = True) -> None:
        # create QApplication if it doesn't already exist
        qapp = get_qapp()

        # Dictionary holding dock widgets
        self._dock_widgets: MutableMapping[str, QtViewerDockWidget] = (
            WeakValueDictionary()
        )
        self._unnamed_dockwidget_count = 1

        self._pref_dialog = None

        # Connect the Viewer and create the Main Window
        self._qt_window = _QtMainWindow(viewer, self)
        qapp.installEventFilter(self._qt_window)

        # connect theme events before collecting plugin-provided themes
        # to ensure icons from the plugins are generated correctly.
        _themes.events.added.connect(self._add_theme)
        _themes.events.removed.connect(self._remove_theme)

        # discover any themes provided by plugins
        plugin_manager.discover_themes()
        self._setup_existing_themes()

        # import and index all discovered shimmed npe1 plugins
        index_npe1_adapters()

        self._add_menus()
        # TODO: the dummy actions should **not** live on the layerlist context
        # as they are unrelated. However, we do not currently have a suitable
        # enclosing context where we could store these keys, such that they
        # **and** the layerlist context key are available when we update
        # menus. We need a single context to contain all keys required for
        # menu update, so we add them to the layerlist context for now.
        add_dummy_actions(self._qt_viewer.viewer.layers._ctx)
        self._update_theme()
        self._update_theme_font_size()
        get_settings().appearance.events.theme.connect(self._update_theme)
        get_settings().appearance.events.font_size.connect(
            self._update_theme_font_size
        )

        self._add_viewer_dock_widget(self._qt_viewer.dockConsole, tabify=False)
        self._add_viewer_dock_widget(
            self._qt_viewer.dockLayerControls,
            tabify=False,
        )
        self._add_viewer_dock_widget(
            self._qt_viewer.dockLayerList, tabify=False
        )
        if perf.perf_config is not None:
            self._add_viewer_dock_widget(
                self._qt_viewer.dockPerformance, menu=self.window_menu
            )

        viewer.events.help.connect(self._help_changed)
        viewer.events.title.connect(self._title_changed)
        viewer.events.theme.connect(self._update_theme)
        viewer.events.status.connect(self._status_changed)

        if show:
            self.show()
            # Ensure the controls dock uses the minimum height
            self._qt_window.resizeDocks(
                [
                    self._qt_viewer.dockLayerControls,
                    self._qt_viewer.dockLayerList,
                ],
                [self._qt_viewer.dockLayerControls.minimumHeight(), 10000],
                Qt.Orientation.Vertical,
            )

    def _setup_existing_themes(self, connect: bool = True):
        """This function is only executed once at the startup of napari
        to connect events to themes that have not been connected yet.

        Parameters
        ----------
        connect : bool
            Determines whether the `connect` or `disconnect` method should be used.
        """
        for theme in _themes.values():
            if connect:
                self._connect_theme(theme)
            else:
                self._disconnect_theme(theme)

    def _connect_theme(self, theme):
        # connect events to update theme. Here, we don't want to pass the event
        # since it won't have the right `value` attribute.
        theme.events.background.connect(self._update_theme_no_event)
        theme.events.foreground.connect(self._update_theme_no_event)
        theme.events.primary.connect(self._update_theme_no_event)
        theme.events.secondary.connect(self._update_theme_no_event)
        theme.events.highlight.connect(self._update_theme_no_event)
        theme.events.text.connect(self._update_theme_no_event)
        theme.events.warning.connect(self._update_theme_no_event)
        theme.events.current.connect(self._update_theme_no_event)
        theme.events.icon.connect(self._update_theme_no_event)
        theme.events.font_size.connect(self._update_theme_no_event)
        theme.events.canvas.connect(
            lambda _: self._qt_viewer.canvas._set_theme_change(
                get_settings().appearance.theme
            )
        )
        # connect console-specific attributes only if QtConsole
        # is present. The `console` is called which might slow
        # things down a little.
        if self._qt_viewer._console:
            theme.events.console.connect(self._qt_viewer.console._update_theme)
            theme.events.syntax_style.connect(
                self._qt_viewer.console._update_theme
            )

    def _disconnect_theme(self, theme):
        theme.events.background.disconnect(self._update_theme_no_event)
        theme.events.foreground.disconnect(self._update_theme_no_event)
        theme.events.primary.disconnect(self._update_theme_no_event)
        theme.events.secondary.disconnect(self._update_theme_no_event)
        theme.events.highlight.disconnect(self._update_theme_no_event)
        theme.events.text.disconnect(self._update_theme_no_event)
        theme.events.warning.disconnect(self._update_theme_no_event)
        theme.events.current.disconnect(self._update_theme_no_event)
        theme.events.icon.disconnect(self._update_theme_no_event)
        theme.events.font_size.disconnect(self._update_theme_no_event)
        theme.events.canvas.disconnect(
            lambda _: self._qt_viewer.canvas._set_theme_change(
                get_settings().appearance.theme
            )
        )
        # disconnect console-specific attributes only if QtConsole
        # is present and they were previously connected
        if self._qt_viewer._console:
            theme.events.console.disconnect(
                self._qt_viewer.console._update_theme
            )
            theme.events.syntax_style.disconnect(
                self._qt_viewer.console._update_theme
            )

    def _add_theme(self, event):
        """Add new theme and connect events."""
        theme = event.value
        self._connect_theme(theme)

    def _remove_theme(self, event):
        """Remove theme and disconnect events."""
        theme = event.value
        self._disconnect_theme(theme)

    @property
    def qt_viewer(self):
        warnings.warn(
            trans._(
                'Public access to Window.qt_viewer is deprecated and will be removed in\n'
                'v0.6.0. It is considered an "implementation detail" of the napari\napplication, '
                'not part of the napari viewer model. If your use case\n'
                'requires access to qt_viewer, please open an issue to discuss.',
                deferred=True,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._qt_window._qt_viewer

    @property
    def _qt_viewer(self):
        # this is starting to be "vestigial"... this property could be removed
        return self._qt_window._qt_viewer

    @property
    def _status_bar(self):
        # TODO: remove from window
        return self._qt_window.statusBar()

    def _update_menu_state(self, menu: MenuStr):
        """Update enabled/visible state of menu item with context."""
        layerlist = self._qt_viewer.viewer.layers
        menu_model = getattr(self, menu)
        menu_model.update_from_context(get_context(layerlist))

    def _update_file_menu_state(self):
        self._update_menu_state('file_menu')

    def _update_view_menu_state(self):
        self._update_menu_state('view_menu')

    def _update_layers_menu_state(self):
        self._update_menu_state('layers_menu')

    def _update_window_menu_state(self):
        self._update_menu_state('window_menu')

    def _update_plugins_menu_state(self):
        self._update_menu_state('plugins_menu')

    def _update_help_menu_state(self):
        self._update_menu_state('help_menu')

    def _update_debug_menu_state(self):
        viewer_ctx = get_context(self._qt_window)
        self._debug_menu.update_from_context(viewer_ctx)

    # TODO: Remove once npe1 deprecated
    def _setup_npe1_samples_menu(self):
        """Register npe1 sample data, build menu and connect to events."""
        plugin_manager.discover_sample_data()
        plugin_manager.events.enabled.connect(_rebuild_npe1_samples_menu)
        plugin_manager.events.disabled.connect(_rebuild_npe1_samples_menu)
        plugin_manager.events.registered.connect(_rebuild_npe1_samples_menu)
        plugin_manager.events.unregistered.connect(_rebuild_npe1_samples_menu)
        _rebuild_npe1_samples_menu()

    # TODO: Remove once npe1 deprecated
    def _setup_npe1_plugins_menu(self):
        """Register npe1 widgets, build menu and connect to events"""
        plugin_manager.discover_widgets()
        plugin_manager.events.registered.connect(_rebuild_npe1_plugins_menu)
        plugin_manager.events.disabled.connect(_rebuild_npe1_plugins_menu)
        plugin_manager.events.unregistered.connect(_rebuild_npe1_plugins_menu)
        _rebuild_npe1_plugins_menu()

    def _handle_trace_file_on_start(self):
        """Start trace of `trace_file_on_start` config set."""
        from napari._qt._qapp_model.qactions._debug import _start_trace

        if perf.perf_config:
            path = perf.perf_config.trace_file_on_start
            if path is not None:
                # Config option "trace_file_on_start" means immediately
                # start tracing to that file. This is very useful if you
                # want to create a trace every time you start napari,
                # without having to start it from the debug menu.
                _start_trace(path)

    def _add_menus(self):
        """Add menubar to napari app."""
        # TODO: move this to _QMainWindow... but then all of the Menu()
        # items will not have easy access to the methods on this Window obj.

        self.main_menu = self._qt_window.menuBar()
        # Menubar shortcuts are only active when the menubar is visible.
        # Therefore, we set a global shortcut not associated with the menubar
        # to toggle visibility, *but*, in order to not shadow the menubar
        # shortcut, we disable it, and only enable it when the menubar is
        # hidden. See this stackoverflow link for details:
        # https://stackoverflow.com/questions/50537642/how-to-keep-the-shortcuts-of-a-hidden-widget-in-pyqt5
        self._main_menu_shortcut = QShortcut('Ctrl+M', self._qt_window)
        self._main_menu_shortcut.setEnabled(False)
        self._main_menu_shortcut.activated.connect(
            self._toggle_menubar_visible
        )
        # file menu
        self.file_menu = build_qmodel_menu(
            MenuId.MENUBAR_FILE, title=trans._('&File'), parent=self._qt_window
        )
        self._setup_npe1_samples_menu()
        self.file_menu.aboutToShow.connect(
            self._update_file_menu_state,
        )
        self.main_menu.addMenu(self.file_menu)
        # view menu
        self.view_menu = build_qmodel_menu(
            MenuId.MENUBAR_VIEW, title=trans._('&View'), parent=self._qt_window
        )
        self.view_menu.aboutToShow.connect(
            self._update_view_menu_state,
        )
        self.main_menu.addMenu(self.view_menu)
        # layers menu
        self.layers_menu = build_qmodel_menu(
            MenuId.MENUBAR_LAYERS,
            title=trans._('&Layers'),
            parent=self._qt_window,
        )
        self.layers_menu.aboutToShow.connect(
            self._update_layers_menu_state,
        )
        self.main_menu.addMenu(self.layers_menu)
        # plugins menu
        self.plugins_menu = build_qmodel_menu(
            MenuId.MENUBAR_PLUGINS,
            title=trans._('&Plugins'),
            parent=self._qt_window,
        )
        self._setup_npe1_plugins_menu()
        self.plugins_menu.aboutToShow.connect(
            self._update_plugins_menu_state,
        )
        self.main_menu.addMenu(self.plugins_menu)
        # debug menu (optional)
        if perf.perf_config is not None:
            self._debug_menu = build_qmodel_menu(
                MenuId.MENUBAR_DEBUG,
                title=trans._('&Debug'),
                parent=self._qt_window,
            )
            self._handle_trace_file_on_start()
            self._debug_menu.aboutToShow.connect(
                self._update_debug_menu_state,
            )
            self.main_menu.addMenu(self._debug_menu)
        # window menu
        self.window_menu = build_qmodel_menu(
            MenuId.MENUBAR_WINDOW,
            title=trans._('&Window'),
            parent=self._qt_window,
        )
        self.plugins_menu.aboutToShow.connect(
            self._update_window_menu_state,
        )
        self.main_menu.addMenu(self.window_menu)
        # help menu
        self.help_menu = build_qmodel_menu(
            MenuId.MENUBAR_HELP, title=trans._('&Help'), parent=self._qt_window
        )
        self.help_menu.aboutToShow.connect(
            self._update_help_menu_state,
        )
        self.main_menu.addMenu(self.help_menu)

    def _toggle_menubar_visible(self):
        """Toggle visibility of app menubar.

        This function also disables or enables a global keyboard shortcut to
        show the menubar, since menubar shortcuts are only available while the
        menubar is visible.
        """
        toggle_menubar_visibility = self._qt_window.toggle_menubar_visibility()
        self._main_menu_shortcut.setEnabled(toggle_menubar_visibility)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self._qt_window.isFullScreen():
            self._qt_window.showNormal()
        else:
            self._qt_window.showFullScreen()

    def _toggle_play(self):
        """Toggle play."""
        if self._qt_viewer.dims.is_playing:
            self._qt_viewer.dims.stop()
        else:
            axis = self._qt_viewer.viewer.dims.last_used or 0
            self._qt_viewer.dims.play(axis)

    def add_plugin_dock_widget(
        self,
        plugin_name: str,
        widget_name: str | None = None,
        tabify: bool = False,
    ) -> tuple[QtViewerDockWidget, Any]:
        """Add plugin dock widget if not already added.

        Parameters
        ----------
        plugin_name : str
            Name of a plugin providing a widget
        widget_name : str, optional
            Name of a widget provided by `plugin_name`. If `None`, and the
            specified plugin provides only a single widget, that widget will be
            returned, otherwise a ValueError will be raised, by default None
        tabify : bool
            Flag to tabify dock widget or not.

        Returns
        -------
        tuple
            A 2-tuple containing (the DockWidget instance, the plugin widget
            instance).
        """
        from napari.plugins import _npe2

        widget_class = None
        dock_kwargs = {}

        if result := _npe2.get_widget_contribution(plugin_name, widget_name):
            widget_class, widget_name = result

        if widget_class is None:
            widget_class, dock_kwargs = plugin_manager.get_widget(
                plugin_name, widget_name
            )

        if not widget_name:
            # if widget_name wasn't provided, `get_widget` will have
            # ensured that there is a single widget available.
            widget_name = next(iter(plugin_manager._dock_widgets[plugin_name]))

        full_name = plugin_menu_item_template.format(plugin_name, widget_name)
        if full_name in self._dock_widgets:
            dock_widget = self._dock_widgets[full_name]
            wdg = dock_widget.widget()
            if hasattr(wdg, '_magic_widget'):
                wdg = wdg._magic_widget
            return dock_widget, wdg

        wdg = _instantiate_dock_widget(
            widget_class, cast('Viewer', self._qt_viewer.viewer)
        )

        # Add dock widget
        dock_kwargs.pop('name', None)
        dock_widget = self.add_dock_widget(
            wdg, name=full_name, tabify=tabify, **dock_kwargs
        )
        return dock_widget, wdg

    def _add_plugin_function_widget(self, plugin_name: str, widget_name: str):
        """Add plugin function widget if not already added.

        Parameters
        ----------
        plugin_name : str
            Name of a plugin providing a widget
        widget_name : str, optional
            Name of a widget provided by `plugin_name`. If `None`, and the
            specified plugin provides only a single widget, that widget will be
            returned, otherwise a ValueError will be raised, by default None
        """
        full_name = plugin_menu_item_template.format(plugin_name, widget_name)
        if full_name in self._dock_widgets:
            return None

        func = plugin_manager._function_widgets[plugin_name][widget_name]

        # Add function widget
        return self.add_function_widget(
            func, name=full_name, area=None, allowed_areas=None
        )

    def add_dock_widget(
        self,
        widget: Union[QWidget, 'Widget'],
        *,
        name: str = '',
        area: str | None = None,
        allowed_areas: Sequence[str] | None = None,
        shortcut=_sentinel,
        add_vertical_stretch=True,
        tabify: bool = False,
        menu: QMenu | None = None,
    ):
        """Convenience method to add a QDockWidget to the main window.

        If name is not provided a generic name will be addded to avoid
        `saveState` warnings on close.

        Parameters
        ----------
        widget : QWidget
            `widget` will be added as QDockWidget's main widget.
        name : str, optional
            Name of dock widget to appear in window menu.
        area : str
            Side of the main window to which the new dock widget will be added.
            Must be in {'left', 'right', 'top', 'bottom'}
        allowed_areas : list[str], optional
            Areas, relative to the main window, that the widget is allowed dock.
            Each item in list must be in {'left', 'right', 'top', 'bottom'}
            By default, all areas are allowed.
        shortcut : str, optional
            Keyboard shortcut to appear in dropdown menu.
        add_vertical_stretch : bool, optional
            Whether to add stretch to the bottom of vertical widgets (pushing
            widgets up towards the top of the allotted area, instead of letting
            them distribute across the vertical space).  By default, True.

            .. deprecated:: 0.4.8

                The shortcut parameter is deprecated since version 0.4.8, please use
                the action and shortcut manager APIs. The new action manager and
                shortcut API allow user configuration and localization.
        tabify : bool
            Flag to tabify dock widget or not.
        menu : QMenu, optional
            Menu bar to add toggle action to. If `None` nothing added to menu.

        Returns
        -------
        dock_widget : QtViewerDockWidget
            `dock_widget` that can pass viewer events.
        """
        if not name:
            with contextlib.suppress(AttributeError):
                name = widget.objectName()
            name = name or trans._(
                'Dock widget {number}',
                number=self._unnamed_dockwidget_count,
            )

            self._unnamed_dockwidget_count += 1

        if area is None:
            settings = get_settings()
            area = settings.application.plugin_widget_positions.get(
                name, 'right'
            )

        if shortcut is not _sentinel:
            warnings.warn(
                _SHORTCUT_DEPRECATION_STRING.format(shortcut=shortcut),
                FutureWarning,
                stacklevel=2,
            )
            dock_widget = QtViewerDockWidget(
                self._qt_viewer,
                widget,
                name=name,
                area=area,
                allowed_areas=allowed_areas,
                shortcut=shortcut,
                add_vertical_stretch=add_vertical_stretch,
            )
        else:
            dock_widget = QtViewerDockWidget(
                self._qt_viewer,
                widget,
                name=name,
                area=area,
                allowed_areas=allowed_areas,
                add_vertical_stretch=add_vertical_stretch,
            )

        self._add_viewer_dock_widget(dock_widget, tabify=tabify, menu=menu)

        if hasattr(widget, 'reset_choices'):
            # Keep the dropdown menus in the widget in sync with the layer model
            # if widget has a `reset_choices`, which is true for all magicgui
            # `CategoricalWidget`s
            layers_events = self._qt_viewer.viewer.layers.events
            layers_events.inserted.connect(widget.reset_choices)
            layers_events.removed.connect(widget.reset_choices)
            layers_events.reordered.connect(widget.reset_choices)

        # Add dock widget to dictionary
        self._dock_widgets[dock_widget.name] = dock_widget

        return dock_widget

    def _add_viewer_dock_widget(
        self,
        dock_widget: QtViewerDockWidget,
        tabify: bool = False,
        menu: QMenu | None = None,
    ):
        """Add a QtViewerDockWidget to the main window

        If other widgets already present in area then will tabify.

        Parameters
        ----------
        dock_widget : QtViewerDockWidget
            `dock_widget` will be added to the main window.
        tabify : bool
            Flag to tabify dockwidget or not.
        menu : QMenu, optional
            Menu bar to add toggle action to. If `None` nothing added to menu.
        """
        # Find if any other dock widgets are currently in area
        current_dws_in_area = [
            dw
            for dw in self._qt_window.findChildren(QDockWidget)
            if self._qt_window.dockWidgetArea(dw) == dock_widget.qt_area
        ]
        self._qt_window.addDockWidget(dock_widget.qt_area, dock_widget)

        # If another dock widget present in area then tabify
        if current_dws_in_area:
            if tabify:
                self._qt_window.tabifyDockWidget(
                    current_dws_in_area[-1], dock_widget
                )
                dock_widget.show()
                dock_widget.raise_()
            elif dock_widget.area in ('right', 'left'):
                _wdg = [*current_dws_in_area, dock_widget]
                # add sizes to push lower widgets up
                sizes = list(range(1, len(_wdg) * 4, 4))
                self._qt_window.resizeDocks(
                    _wdg, sizes, Qt.Orientation.Vertical
                )

        if menu:
            action = dock_widget.toggleViewAction()
            action.setStatusTip(dock_widget.name)
            action.setText(dock_widget.name)
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                # deprecating with 0.4.8, but let's try to keep compatibility.
                shortcut = dock_widget.shortcut
            if shortcut is not None:
                action.setShortcut(shortcut)

            menu.addAction(action)

        # see #3663, to fix #3624 more generally
        dock_widget.setFloating(False)

    def _remove_dock_widget(self, event) -> None:
        names = list(self._dock_widgets.keys())
        for widget_name in names:
            if event.value in widget_name:
                # remove this widget
                widget = self._dock_widgets[widget_name]
                self.remove_dock_widget(widget)

    def remove_dock_widget(self, widget: QWidget, menu=None):
        """Removes specified dock widget.

        If a QDockWidget is not provided, the existing QDockWidgets will be
        searched for one whose inner widget (``.widget()``) is the provided
        ``widget``.

        Parameters
        ----------
        widget : QWidget | str
            If widget == 'all', all docked widgets will be removed.
        menu : QMenu, optional
            Menu bar to remove toggle action from. If `None` nothing removed
            from menu.
        """
        if widget == 'all':
            for dw in list(self._dock_widgets.values()):
                self.remove_dock_widget(dw)
            return

        if not isinstance(widget, QDockWidget):
            dw: QDockWidget
            for dw in self._qt_window.findChildren(QDockWidget):
                if dw.widget() is widget:
                    _dw: QDockWidget = dw
                    break
            else:
                raise LookupError(
                    trans._(
                        'Could not find a dock widget containing: {widget}',
                        deferred=True,
                        widget=widget,
                    )
                )
        else:
            _dw = widget

        if _dw.widget():
            _dw.widget().setParent(None)
        self._qt_window.removeDockWidget(_dw)
        if menu is not None:
            menu.removeAction(_dw.toggleViewAction())

        # Remove dock widget from dictionary
        self._dock_widgets.pop(_dw.name, None)

        # Deleting the dock widget means any references to it will no longer
        # work but it's not really useful anyway, since the inner widget has
        # been removed. and anyway: people should be using add_dock_widget
        # rather than directly using _add_viewer_dock_widget
        _dw.deleteLater()

    def add_function_widget(
        self,
        function,
        *,
        magic_kwargs=None,
        name: str = '',
        area=None,
        allowed_areas=None,
        shortcut=_sentinel,
    ):
        """Turn a function into a dock widget via magicgui.

        Parameters
        ----------
        function : callable
            Function that you want to add.
        magic_kwargs : dict, optional
            Keyword arguments to :func:`magicgui.magicgui` that
            can be used to specify widget.
        name : str, optional
            Name of dock widget to appear in window menu.
        area : str, optional
            Side of the main window to which the new dock widget will be added.
            Must be in {'left', 'right', 'top', 'bottom'}. If not provided the
            default will be determined by the widget.layout, with 'vertical'
            layouts appearing on the right, otherwise on the bottom.
        allowed_areas : list[str], optional
            Areas, relative to main window, that the widget is allowed dock.
            Each item in list must be in {'left', 'right', 'top', 'bottom'}
            By default, only provided areas is allowed.
        shortcut : str, optional
            Keyboard shortcut to appear in dropdown menu.

        Returns
        -------
        dock_widget : QtViewerDockWidget
            `dock_widget` that can pass viewer events.
        """
        from magicgui import magicgui

        if magic_kwargs is None:
            magic_kwargs = {
                'auto_call': False,
                'call_button': 'run',
                'layout': 'vertical',
            }

        widget = magicgui(function, **magic_kwargs or {})

        if area is None:
            area = 'right' if str(widget.layout) == 'vertical' else 'bottom'
        if allowed_areas is None:
            allowed_areas = [area]
        if shortcut is not _sentinel:
            return self.add_dock_widget(
                widget,
                name=name or function.__name__.replace('_', ' '),
                area=area,
                allowed_areas=allowed_areas,
                shortcut=shortcut,
            )

        return self.add_dock_widget(
            widget,
            name=name or function.__name__.replace('_', ' '),
            area=area,
            allowed_areas=allowed_areas,
        )

    def resize(self, width, height):
        """Resize the window.

        Parameters
        ----------
        width : int
            Width in logical pixels.
        height : int
            Height in logical pixels.
        """
        self._qt_window.resize(width, height)

    def set_geometry(self, left, top, width, height):
        """Set the geometry of the widget

        Parameters
        ----------
        left : int
            X coordinate of the upper left border.
        top : int
            Y coordinate of the upper left border.
        width : int
            Width of the rectangle shape of the window.
        height : int
            Height of the rectangle shape of the window.
        """
        self._qt_window.setGeometry(left, top, width, height)

    def geometry(self) -> tuple[int, int, int, int]:
        """Get the geometry of the widget

        Returns
        -------
        left : int
            X coordinate of the upper left border.
        top : int
            Y coordinate of the upper left border.
        width : int
            Width of the rectangle shape of the window.
        height : int
            Height of the rectangle shape of the window.
        """
        rect = self._qt_window.geometry()
        return rect.left(), rect.top(), rect.width(), rect.height()

    def show(self, *, block=False):
        """Resize, show, and bring forward the window.

        Raises
        ------
        RuntimeError
            If the viewer.window has already been closed and deleted.
        """
        settings = get_settings()
        try:
            self._qt_window.show(block=block)
        except (AttributeError, RuntimeError) as e:
            raise RuntimeError(
                trans._(
                    'This viewer has already been closed and deleted. Please create a new one.',
                    deferred=True,
                )
            ) from e

        if settings.application.first_time:
            settings.application.first_time = False
            try:
                self._qt_window.resize(self._qt_window.layout().sizeHint())
            except (AttributeError, RuntimeError) as e:
                raise RuntimeError(
                    trans._(
                        'This viewer has already been closed and deleted. Please create a new one.',
                        deferred=True,
                    )
                ) from e
        else:
            try:
                if settings.application.save_window_geometry:
                    self._qt_window._set_window_settings(
                        *self._qt_window._load_window_settings()
                    )
            except Exception as err:  # noqa: BLE001
                import warnings

                warnings.warn(
                    trans._(
                        'The window geometry settings could not be loaded due to the following error: {err}',
                        deferred=True,
                        err=err,
                    ),
                    category=RuntimeWarning,
                    stacklevel=2,
                )

        # Resize axis labels now that window is shown
        self._qt_viewer.dims._resize_axis_labels()

        # We want to bring the viewer to the front when
        # A) it is our own event loop OR we are running in jupyter
        # B) it is not the first time a QMainWindow is being created

        # `app_name` will be "napari" iff the application was instantiated in
        # get_qapp(). isActiveWindow() will be True if it is the second time a
        # _qt_window has been created.
        # See #721, #732, #735, #795, #1594
        app_name = QApplication.instance().applicationName()
        if (
            app_name == 'napari' or in_jupyter()
        ) and self._qt_window.isActiveWindow():
            self.activate()

    def activate(self):
        """Make the viewer the currently active window."""
        self._qt_window.raise_()  # for macOS
        self._qt_window.activateWindow()  # for Windows

    def _update_theme_no_event(self):
        self._update_theme()

    def _update_theme_font_size(self, event=None):
        settings = get_settings()
        font_size = event.value if event else settings.appearance.font_size
        extra_variables = {'font_size': f'{font_size}pt'}
        self._update_theme(extra_variables=extra_variables)

    def _update_theme(self, event=None, extra_variables=None):
        """Update widget color theme."""
        if extra_variables is None:
            extra_variables = {}
        settings = get_settings()
        with contextlib.suppress(AttributeError, RuntimeError):
            value = event.value if event else settings.appearance.theme
            self._qt_viewer.viewer.theme = value
            actual_theme_name = value
            if value == 'system':
                # system isn't a theme, so get the name
                actual_theme_name = get_system_theme()
            # check `font_size` value is always passed when updating style
            if 'font_size' not in extra_variables:
                extra_variables.update(
                    {'font_size': f'{settings.appearance.font_size}pt'}
                )
            # set the style sheet with the theme name and extra_variables
            style_sheet = get_stylesheet(
                actual_theme_name, extra_variables=extra_variables
            )
            self._qt_window.setStyleSheet(style_sheet)
            self._qt_viewer.setStyleSheet(style_sheet)
            if self._qt_viewer._console:
                self._qt_viewer._console._update_theme(style_sheet=style_sheet)

    def _status_changed(self, event):
        """Update status bar.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if isinstance(event.value, str):
            self._status_bar.setStatusText(event.value)
        else:
            status_info = event.value
            self._status_bar.setStatusText(
                layer_base=status_info['layer_base'],
                source_type=status_info['source_type'],
                plugin=status_info['plugin'],
                coordinates=status_info['coordinates'],
            )

    def _title_changed(self, event):
        """Update window title.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self._qt_window.setWindowTitle(event.value)

    def _help_changed(self, event):
        """Update help message on status bar.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self._status_bar.setHelpText(event.value)

    def _restart(self):
        """Restart the napari application."""
        self._qt_window.restart()

    def _screenshot(
        self,
        size: tuple[int, int] | None = None,
        scale: float | None = None,
        flash: bool = True,
        canvas_only: bool = False,
        fit_to_data_extent: bool = False,
    ) -> 'QImage':
        """Capture screenshot of the currently displayed viewer.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        size : tuple of two ints, optional
            Size (resolution height x width) of the screenshot. By default, the
            currently displayed size. Only used if `canvas_only` is True. This
            argument is ignored if fit_to_data_extent is set to True.
        scale : float, optional
            Scale factor used to increase resolution of canvas for the screenshot.
            By default, the currently displayed resolution.
            Only used if `canvas_only` is True.
        canvas_only : bool
            If True, screenshot shows only the image display canvas, and
            if False include the napari viewer frame in the screenshot,
            By default, True.
        fit_to_data_extent: bool
            Tightly fit the canvas around the data to prevent margins from
            showing in the screenshot. If False, a screenshot of the currently
            visible canvas will be generated.

        Returns
        -------
        img : QImage
        """
        from napari._qt.utils import add_flash_animation

        canvas = self._qt_viewer.canvas
        prev_size = canvas.size
        camera = self._qt_viewer.viewer.camera
        old_center = camera.center
        old_zoom = camera.zoom
        ndisplay = self._qt_viewer.viewer.dims.ndisplay

        # Part 1: validate incompatible parameters
        if not canvas_only and (
            fit_to_data_extent or size is not None or scale is not None
        ):
            raise ValueError(
                trans._(
                    'scale, size, and fit_to_data_extent can only be set for '
                    'canvas_only screenshots.',
                    deferred=True,
                )
            )
        if fit_to_data_extent and ndisplay > 2:
            raise NotImplementedError(
                trans._(
                    'fit_to_data_extent is not yet implemented for 3D view.',
                    deferred=True,
                )
            )
        if size is not None and len(size) != 2:
            raise ValueError(
                trans._(
                    'screenshot size must be 2 values, got {len_size}',
                    deferred=True,
                    len_size=len(size),
                )
            )

        # Part 2: compute canvas size and view based on parameters
        if fit_to_data_extent:
            extent_world = self._qt_viewer.viewer.layers.extent.world[1][
                -ndisplay:
            ]
            extent_step = min(
                self._qt_viewer.viewer.layers.extent.step[-ndisplay:]
            )
            size = extent_world / extent_step + 1
        if size is not None:
            size = np.asarray(size) / self._qt_window.devicePixelRatio()
        else:
            size = np.asarray(prev_size)
        if scale is not None:
            # multiply canvas dimensions by the scale factor to get new size
            size *= scale

        # Part 3: take the screenshot
        if canvas_only:
            canvas.size = tuple(size.astype(int))
            if fit_to_data_extent:
                # tight view around data
                self._qt_viewer.viewer.reset_view(margin=0)
            try:
                img = canvas.screenshot()
                if flash:
                    add_flash_animation(self._qt_viewer._welcome_widget)
            finally:
                # make sure we always go back to the right canvas size
                canvas.size = prev_size
                camera.center = old_center
                camera.zoom = old_zoom
        else:
            img = self._qt_window.grab().toImage()
            if flash:
                add_flash_animation(self._qt_window)
        return img

    def export_figure(
        self,
        path: str | None = None,
        scale: float = 1,
        flash=True,
    ) -> np.ndarray:
        """Export an image of the full extent of the displayed layer data.

        This function finds a tight boundary around the data, resets the view
        around that boundary (and, when scale=1, such that 1 captured pixel is
        equivalent to one data pixel), takes a screenshot, then restores the
        previous zoom and canvas sizes. Currently, only works when 2 dimensions
        are displayed.

        Parameters
        ----------
        path : str, optional
            Filename for saving screenshot image.
        scale : float
            Scale factor used to increase resolution of canvas for the
            screenshot. By default, a scale of 1.
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
            By default, True.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        if not isinstance(scale, (float, int)):
            raise TypeError(
                trans._(
                    'Scale must be a float or an int.',
                    deferred=True,
                )
            )
        img = QImg2array(
            self._screenshot(
                scale=scale,
                flash=flash,
                canvas_only=True,
                fit_to_data_extent=True,
            )
        )
        if path is not None:
            imsave(path, img)
        return img

    def screenshot(
        self, path=None, size=None, scale=None, flash=True, canvas_only=False
    ):
        """Take currently displayed viewer and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.
        size : tuple (int, int)
            Size (resolution) of the screenshot. By default, the currently displayed size.
            Only used if `canvas_only` is True.
        scale : float
            Scale factor used to increase resolution of canvas for the screenshot.
            By default, the currently displayed resolution.
            Only used if `canvas_only` is True.
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        canvas_only : bool
            If True, screenshot shows only the image display canvas, and
            if False includes the napari viewer frame in the screenshot,
            By default, True.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """

        img = QImg2array(self._screenshot(size, scale, flash, canvas_only))
        if path is not None:
            imsave(path, img)
        return img

    def clipboard(self, flash=True, canvas_only=False):
        """Copy screenshot of current viewer to the clipboard.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        canvas_only : bool
            If True, screenshot shows only the image display canvas, and
            if False include the napari viewer frame in the screenshot,
            By default, True.
        """
        img = self._screenshot(flash=flash, canvas_only=canvas_only)
        QApplication.clipboard().setImage(img)

    def _teardown(self):
        """Carry out various teardown tasks such as event disconnection."""
        self._setup_existing_themes(False)
        _themes.events.added.disconnect(self._add_theme)
        _themes.events.removed.disconnect(self._remove_theme)

    def close(self):
        """Close the viewer window and cleanup sub-widgets."""
        # Someone is closing us twice? Only try to delete self._qt_window
        # if we still have one.
        if hasattr(self, '_qt_window'):
            self._teardown()
            self._qt_viewer.close()
            self._qt_window.close()
            del self._qt_window

    def _open_preferences_dialog(self) -> PreferencesDialog:
        """Edit preferences from the menubar."""
        if self._pref_dialog is None:
            win = PreferencesDialog(parent=self._qt_window)
            self._pref_dialog = win

            app_pref = get_settings().application
            if app_pref.preferences_size:
                win.resize(*app_pref.preferences_size)

            @win.resized.connect
            def _save_size(sz: QSize):
                app_pref.preferences_size = (sz.width(), sz.height())

            def _clean_pref_dialog():
                self._pref_dialog = None

            win.finished.connect(_clean_pref_dialog)
            win.show()
        else:
            self._pref_dialog.raise_()

        return self._pref_dialog

    def _screenshot_dialog(self):
        """Save screenshot of current display with viewer, default .png"""
        from napari._qt.dialogs.screenshot_dialog import ScreenshotDialog
        from napari.utils.history import get_save_history, update_save_history

        hist = get_save_history()
        dial = ScreenshotDialog(
            self.screenshot, self._qt_viewer, hist[0], hist
        )
        if dial.exec_():
            update_save_history(dial.selectedFiles()[0])


def _instantiate_dock_widget(wdg_cls, viewer: 'Viewer'):
    # if the signature is looking a for a napari viewer, pass it.
    from napari.viewer import Viewer

    kwargs = {}
    try:
        sig = inspect.signature(wdg_cls.__init__)
    # Inspection can fail when adding to bundle as it thinks widget is a builtin
    except ValueError:
        pass
    else:
        for param in sig.parameters.values():
            if param.name == 'napari_viewer':
                kwargs['napari_viewer'] = PublicOnlyProxy(viewer)
                break
            if param.annotation in ('napari.viewer.Viewer', Viewer):
                kwargs[param.name] = PublicOnlyProxy(viewer)
                break
            # cannot look for param.kind == param.VAR_KEYWORD because
            # QWidget allows **kwargs but errs on unknown keyword arguments

    # instantiate the widget
    return wdg_cls(**kwargs)
