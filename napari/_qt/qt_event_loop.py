from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast
from warnings import warn

from qtpy import PYQT5, PYSIDE2
from qtpy.QtCore import QDir, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QWidget

from napari import Viewer, __version__
from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari._qt.qt_event_filters import QtToolTipEventFilter
from napari._qt.qthreading import (
    register_threadworker_processors,
    wait_for_workers_to_quit,
)
from napari._qt.utils import _maybe_allow_interrupt
from napari.resources._icons import _theme_path
from napari.settings import get_settings
from napari.utils import config, perf
from napari.utils.notifications import (
    notification_manager,
    show_console_notification,
)
from napari.utils.perf import perf_config
from napari.utils.theme import _themes
from napari.utils.translations import trans

if TYPE_CHECKING:
    from IPython import InteractiveShell

NAPARI_ICON_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'resources', 'logo.png'
)
NAPARI_APP_ID = f'napari.napari.viewer.{__version__}'


def set_app_id(app_id):
    if os.name == 'nt' and app_id and not getattr(sys, 'frozen', False):
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)


_defaults = {
    'app_name': 'napari',
    'app_version': __version__,
    'icon': NAPARI_ICON_PATH,
    'org_name': 'napari',
    'org_domain': 'napari.org',
    'app_id': NAPARI_APP_ID,
}


# store reference to QApplication to prevent garbage collection
_app_ref = None
_IPYTHON_WAS_HERE_FIRST = 'IPython' in sys.modules


def _focus_changed(old: QWidget | None, new: QWidget | None):
    if old is not None and new is not None:
        return  # ignore focus changes between two widgets
    if old is None and new is None:
        return  # this should not happen
    if old is None:
        start_timer = True
        window = new.window()
    else:
        start_timer = False
        window = old.window()

    from napari._qt.qt_main_window import _QtMainWindow

    if not isinstance(window, _QtMainWindow):
        return
    notifications = cast(
        list[NapariQtNotification], window.findChildren(NapariQtNotification)
    )
    if not notifications:
        return
    if start_timer:
        notifications[-1].timer_start()
    else:
        for notification in notifications:
            notification.timer_stop()


# TODO: Remove in napari 0.6.0
def get_app(*args, **kwargs) -> QApplication:
    """Get or create the Qt QApplication. Now deprecated, use `get_qapp`."""
    warn(
        trans._(
            '`QApplication` instance access through `get_app` is deprecated and will be removed in 0.6.0.\n'
            'Please use `get_qapp` instead.\n',
            deferred=True,
        ),
        category=FutureWarning,
        stacklevel=2,
    )
    return get_qapp(*args, **kwargs)


def get_qapp(
    *,
    app_name: str | None = None,
    app_version: str | None = None,
    icon: str | None = None,
    org_name: str | None = None,
    org_domain: str | None = None,
    app_id: str | None = None,
    ipy_interactive: bool | None = None,
) -> QApplication:
    """Get or create the Qt QApplication.

    There is only one global QApplication instance, which can be retrieved by
    calling get_app again, (or by using QApplication.instance())

    Parameters
    ----------
    app_name : str, optional
        Set app name (if creating for the first time), by default 'napari'
    app_version : str, optional
        Set app version (if creating for the first time), by default __version__
    icon : str, optional
        Set app icon (if creating for the first time), by default
        NAPARI_ICON_PATH
    org_name : str, optional
        Set organization name (if creating for the first time), by default
        'napari'
    org_domain : str, optional
        Set organization domain (if creating for the first time), by default
        'napari.org'
    app_id : str, optional
        Set organization domain (if creating for the first time).  Will be
        passed to set_app_id (which may also be called independently), by
        default NAPARI_APP_ID
    ipy_interactive : bool, optional
        Use the IPython Qt event loop ('%gui qt' magic) if running in an
        interactive IPython terminal.

    Returns
    -------
    QApplication
        [description]

    Notes
    -----
    Substitutes QApplicationWithTracing when the NAPARI_PERFMON env variable
    is set.

    """
    # napari defaults are all-or nothing.  If any of the keywords are used
    # then they are all used.
    set_values = {k for k, v in locals().items() if v}
    kwargs = locals() if set_values else _defaults
    global _app_ref

    app = QApplication.instance()
    if app:
        set_values.discard('ipy_interactive')
        if set_values:
            warn(
                trans._(
                    "QApplication already existed, these arguments to to 'get_app' were ignored: {args}",
                    deferred=True,
                    args=set_values,
                ),
                stacklevel=2,
            )
        if perf_config and perf_config.trace_qt_events:
            warn(
                trans._(
                    'Using NAPARI_PERFMON with an already-running QtApp (--gui qt?) is not supported.',
                    deferred=True,
                ),
                stacklevel=2,
            )

    else:
        # automatically determine monitor DPI.
        # Note: this MUST be set before the QApplication is instantiated. Also, this
        # attributes need to be applied only to Qt5 bindings (PyQt5 and PySide2)
        # since the High DPI scaling attributes are deactivated by default while on Qt6
        # they are deprecated and activated by default. For more info see:
        # https://doc.qt.io/qtforpython-6/gettingstarted/porting_from2.html#class-function-deprecations
        if PYQT5 or PYSIDE2:
            QApplication.setAttribute(
                Qt.ApplicationAttribute.AA_EnableHighDpiScaling
            )
            QApplication.setAttribute(
                Qt.ApplicationAttribute.AA_UseHighDpiPixmaps
            )

        argv = sys.argv.copy()
        if sys.platform == 'darwin' and not argv[0].endswith('napari'):
            # Make sure the app name in the Application menu is `napari`
            # which is taken from the basename of sys.argv[0]; we use
            # a copy so the original value is still available at sys.argv
            argv[0] = 'napari'

        if perf_config and perf_config.trace_qt_events:
            from napari._qt.perf.qt_event_tracing import (
                QApplicationWithTracing,
            )

            app = QApplicationWithTracing(argv)
        else:
            app = QApplication(argv)

        # if this is the first time the Qt app is being instantiated, we set
        # the name and metadata
        app.setApplicationName(kwargs.get('app_name'))
        app.setApplicationVersion(kwargs.get('app_version'))
        app.setOrganizationName(kwargs.get('org_name'))
        app.setOrganizationDomain(kwargs.get('org_domain'))
        set_app_id(kwargs.get('app_id'))

        # Intercept tooltip events in order to convert all text to rich text
        # to allow for text wrapping of tooltips
        app.installEventFilter(QtToolTipEventFilter())

    if app.windowIcon().isNull():
        app.setWindowIcon(QIcon(kwargs.get('icon')))

    if ipy_interactive is None:
        ipy_interactive = get_settings().application.ipy_interactive
    if _IPYTHON_WAS_HERE_FIRST:
        _try_enable_ipython_gui('qt' if ipy_interactive else None)

    if perf_config and not perf_config.patched:
        # Will patch based on config file.
        perf_config.patch_callables()

    if not _app_ref:  # running get_app for the first time
        # see docstring of `wait_for_workers_to_quit` for caveats on killing
        # workers at shutdown.
        app.aboutToQuit.connect(wait_for_workers_to_quit)

        # Setup search paths for currently installed themes.
        for name in _themes:
            QDir.addSearchPath(f'theme_{name}', str(_theme_path(name)))

        # When a new theme is added, at it to the search path.
        @_themes.events.changed.connect
        @_themes.events.added.connect
        def _(event):
            name = event.key
            QDir.addSearchPath(f'theme_{name}', str(_theme_path(name)))

        register_threadworker_processors()

        notification_manager.notification_ready.connect(
            NapariQtNotification.show_notification
        )
        notification_manager.notification_ready.connect(
            show_console_notification
        )

        app.focusChanged.connect(_focus_changed)

    _app_ref = app  # prevent garbage collection

    # Add the dispatcher attribute to the application to be able to dispatch
    # notifications coming from threads

    return app


def quit_app():
    """Close all windows and quit the QApplication if napari started it."""
    for v in list(Viewer._instances):
        v.close()
    QApplication.closeAllWindows()
    # if we started the application then the app will be named 'napari'.
    if (
        QApplication.applicationName() == 'napari'
        and not _ipython_has_eventloop()
    ):
        QApplication.quit()

    # otherwise, something else created the QApp before us (such as
    # %gui qt IPython magic).  If we quit the app in this case, then
    # *later* attempts to instantiate a napari viewer won't work until
    # the event loop is restarted with app.exec_().  So rather than
    # quit just close all the windows (and clear our app icon).
    else:
        QApplication.setWindowIcon(QIcon())

    if perf.perf_config is not None:
        # Write trace file before exit, if we were writing one.
        # Is there a better place to make sure this is done on exit?
        perf.timers.stop_trace_file()

    if config.monitor:
        # Stop the monitor service if we were using it
        from napari.components.experimental.monitor import monitor

        monitor.stop()


@contextmanager
def gui_qt(*, startup_logo=False, gui_exceptions=False, force=False):
    """Start a Qt event loop in which to run the application.

    NOTE: This context manager is deprecated!. Prefer using :func:`napari.run`.

    Parameters
    ----------
    startup_logo : bool, optional
        Show a splash screen with the napari logo during startup.
    gui_exceptions : bool, optional
        Whether to show uncaught exceptions in the GUI, by default they will be
        shown in the console that launched the event loop.
    force : bool, optional
        Force the application event_loop to start, even if there are no top
        level widgets to show.

    Notes
    -----
    This context manager is not needed if running napari within an interactive
    IPython session. In this case, use the ``%gui qt`` magic command, or start
    IPython with the Qt GUI event loop enabled by default by using
    ``ipython --gui=qt``.
    """
    warn(
        trans._(
            "\nThe 'gui_qt()' context manager is deprecated.\nIf you are running napari from a script, please use 'napari.run()' as follows:\n\n    import napari\n\n    viewer = napari.Viewer()  # no prior setup needed\n    # other code using the viewer...\n    napari.run()\n\nIn IPython or Jupyter, 'napari.run()' is not necessary. napari will automatically\nstart an interactive event loop for you: \n\n    import napari\n    viewer = napari.Viewer()  # that's it!\n",
            deferred=True,
        ),
        FutureWarning,
        stacklevel=2,
    )

    app = get_app()
    splash = None
    if startup_logo and app.applicationName() == 'napari':
        from napari._qt.widgets.qt_splash_screen import NapariSplashScreen

        splash = NapariSplashScreen()
        splash.close()
    try:
        yield app
    except Exception:  # noqa: BLE001
        notification_manager.receive_error(*sys.exc_info())
    run(force=force, gui_exceptions=gui_exceptions, _func_name='gui_qt')


def _ipython_has_eventloop() -> bool:
    """Return True if IPython %gui qt is active.

    Using this is better than checking ``QApp.thread().loopLevel() > 0``,
    because IPython starts and stops the event loop continuously to accept code
    at the prompt.  So it will likely "appear" like there is no event loop
    running, but we still don't need to start one.
    """
    ipy_module = sys.modules.get('IPython')
    if not ipy_module:
        return False

    shell: InteractiveShell = ipy_module.get_ipython()  # type: ignore
    if not shell:
        return False

    return shell.active_eventloop == 'qt'


def _pycharm_has_eventloop(app: QApplication) -> bool:
    """Return true if running in PyCharm and eventloop is active.

    Explicit checking is necessary because PyCharm runs a custom interactive
    shell which overrides `InteractiveShell.enable_gui()`, breaking some
    superclass behaviour.
    """
    in_pycharm = 'PYCHARM_HOSTED' in os.environ
    in_event_loop = getattr(app, '_in_event_loop', False)
    return in_pycharm and in_event_loop


def _try_enable_ipython_gui(gui='qt'):
    """Start %gui qt the eventloop."""
    ipy_module = sys.modules.get('IPython')
    if not ipy_module:
        return

    shell: InteractiveShell = ipy_module.get_ipython()  # type: ignore
    if not shell:
        return
    if shell.active_eventloop != gui:
        shell.enable_gui(gui)


def run(
    *, force=False, gui_exceptions=False, max_loop_level=1, _func_name='run'
):
    """Start the Qt Event Loop

    Parameters
    ----------
    force : bool, optional
        Force the application event_loop to start, even if there are no top
        level widgets to show.
    gui_exceptions : bool, optional
        Whether to show uncaught exceptions in the GUI. By default they will be
        shown in the console that launched the event loop.
    max_loop_level : int, optional
        The maximum allowable "loop level" for the execution thread.  Every
        time `QApplication.exec_()` is called, Qt enters the event loop,
        increments app.thread().loopLevel(), and waits until exit() is called.
        This function will prevent calling `exec_()` if the application already
        has at least ``max_loop_level`` event loops running.  By default, 1.
    _func_name : str, optional
        name of calling function, by default 'run'.  This is only here to
        provide functions like `gui_qt` a way to inject their name into the
        warning message.

    Raises
    ------
    RuntimeError
        (To avoid confusion) if no widgets would be shown upon starting the
        event loop.
    """
    if _ipython_has_eventloop():
        # If %gui qt is active, we don't need to block again.
        return

    app = QApplication.instance()

    if _pycharm_has_eventloop(app):
        # explicit check for PyCharm pydev console
        return

    if not app:
        raise RuntimeError(
            trans._(
                'No Qt app has been created. One can be created by calling `get_app()` or `qtpy.QtWidgets.QApplication([])`',
                deferred=True,
            )
        )
    if not app.topLevelWidgets() and not force:
        warn(
            trans._(
                'Refusing to run a QApplication with no topLevelWidgets. To run the app anyway, use `{_func_name}(force=True)`',
                deferred=True,
                _func_name=_func_name,
            ),
            stacklevel=2,
        )
        return

    if app.thread().loopLevel() >= max_loop_level:
        loops = app.thread().loopLevel()
        warn(
            trans._n(
                'A QApplication is already running with 1 event loop. To enter *another* event loop, use `{_func_name}(max_loop_level={max_loop_level})`',
                'A QApplication is already running with {n} event loops. To enter *another* event loop, use `{_func_name}(max_loop_level={max_loop_level})`',
                n=loops,
                deferred=True,
                _func_name=_func_name,
                max_loop_level=loops + 1,
            ),
            stacklevel=2,
        )
        return
    with notification_manager, _maybe_allow_interrupt(app):
        app.exec_()
