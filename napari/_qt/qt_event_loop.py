import os
import sys
from contextlib import contextmanager
from warnings import warn

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication

from napari import __version__
from napari.utils.notifications import notification_manager

from ..utils import config, perf
from ..utils.perf import perf_config
from .exceptions import ExceptionHandler
from .qt_resources import _register_napari_resources
from .dialogs.qt_notification import NapariQtNotification
from .qthreading import wait_for_workers_to_quit

NAPARI_ICON_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'resources', 'logo.png'
)
NAPARI_APP_ID = f'napari.napari.viewer.{__version__}'


def set_app_id(app_id):
    if os.name == "nt" and app_id and not getattr(sys, 'frozen', False):
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


def get_app(
    *,
    app_name: str = None,
    app_version: str = None,
    icon: str = None,
    org_name: str = None,
    org_domain: str = None,
    app_id: str = None,
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

    Returns
    -------
    QApplication
        [description]

    Notes
    -----
    Substitutes QApplicationWithTracing when the NAPARI_PERFMON env variable
    is set.

    If the QApplication already exists, we call convert_app_for_tracing() which
    deletes the QApplication and creates a new one. However here with get_app
    we need to create the correct QApplication up front, or we will crash
    because we'd be deleting the QApplication after we created QWidgets with
    it, such as we do for the splash screen.
    """
    # napari defaults are all-or nothing.  If any of the keywords are used
    # then they are all used.
    set_values = {k for k, v in locals().items() if v}
    kwargs = locals() if set_values else _defaults
    global _app_ref

    app = QApplication.instance()
    if app:
        if set_values:

            warn(
                "QApplication already existed, these arguments to to 'get_app'"
                " were ignored: {}".format(set_values)
            )
        if perf_config and perf_config.trace_qt_events:
            from .perf.qt_event_tracing import convert_app_for_tracing

            # no-op if app is already a QApplicationWithTracing
            app = convert_app_for_tracing(app)
    else:
        # automatically determine monitor DPI.
        # Note: this MUST be set before the QApplication is instantiated
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

        if perf_config and perf_config.trace_qt_events:
            from .perf.qt_event_tracing import QApplicationWithTracing

            app = QApplicationWithTracing(sys.argv)
        else:
            app = QApplication(sys.argv)

        # if this is the first time the Qt app is being instantiated, we set
        # the name and metadata
        app.setApplicationName(kwargs.get('app_name'))
        app.setApplicationVersion(kwargs.get('app_version'))
        app.setOrganizationName(kwargs.get('org_name'))
        app.setOrganizationDomain(kwargs.get('org_domain'))
        app.setWindowIcon(QIcon(kwargs.get('icon')))
        set_app_id(kwargs.get('app_id'))

        if os.getenv("NAPARI_CATCH_ERRORS") not in ('0', 'False'):

            app.arena = []

            def show_notification(*args, **kwargs):
                notif = NapariQtNotification.from_notification(*args, **kwargs)
                notif.show()
                app.arena.append(notif)

            notification_manager.notification_ready.connect(show_notification)

    if perf_config and not perf_config.patched:
        # Will patch based on config file.
        perf_config.patch_callables()

    if not _app_ref:  # running get_app for the first time
        # see docstring of `wait_for_workers_to_quit` for caveats on killing
        # workers at shutdown.
        app.aboutToQuit.connect(wait_for_workers_to_quit)

        # this will register all of our resources (icons) with Qt, so that they
        # can be used in qss files and elsewhere.
        _register_napari_resources()

    _app_ref = app  # prevent garbage collection
    return app


def quit_app():
    """Close all windows and quit the QApplication if napari started it."""
    QApplication.closeAllWindows()
    # if we started the application then the app will be named 'napari'.
    if QApplication.applicationName() == 'napari':
        QApplication.quit()

    # otherwise, something else created the QApp before us (such as
    # %gui qt IPython magic).  If we quit the app in this case, then
    # *later* attempts to instantiate a napari viewer won't work until
    # the event loop is restarted with app.exec_().  So rather than
    # quit just close all the windows (and clear our app icon).
    else:
        QApplication.setWindowIcon(QIcon())

    if perf.USE_PERFMON:
        # Write trace file before exit, if we were writing one.
        # Is there a better place to make sure this is done on exit?
        perf.timers.stop_trace_file()

    if config.monitor:
        # Stop the monitor service if we were using it
        from ..components.experimental.monitor import monitor

        monitor.stop()

    if config.async_loading:
        # Shutdown the chunkloader
        from ..components.experimental.chunk import chunk_loader

        chunk_loader.shutdown()


@contextmanager
def gui_qt(*, startup_logo=False, gui_exceptions=False, force=False):
    """Start a Qt event loop in which to run the application.

    NOTE: This context manager may be deprecated in the future. Prefer using
    :func:`napari.run` instead.

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

    app = get_app()
    splash = None
    if startup_logo and app.applicationName() == 'napari':
        from .widgets.qt_splash_screen import NapariSplashScreen

        splash = NapariSplashScreen()
        splash.close()
    try:
        yield app
    except Exception as e:
        notification_manager.from_exception(e)
    run(force=force, gui_exceptions=gui_exceptions, _func_name='gui_qt')


def _ipython_has_eventloop() -> bool:
    """Return True if IPython %gui qt is active.

    Using this is better than checking ``QApp.thread().loopLevel() > 0``,
    because IPython starts and stops the event loop continuously to accept code
    at the prompt.  So it will likely "appear" like there is no event loop
    running, but we still don't need to start one.
    """
    try:
        from IPython import get_ipython

        return get_ipython().active_eventloop == 'qt'
    except (ImportError, AttributeError):
        return False


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
    if not app:
        raise RuntimeError(
            'No Qt app has been created. '
            'One can be created by calling `get_app()` '
            'or qtpy.QtWidgets.QApplication([])'
        )
    if not app.topLevelWidgets() and not force:
        warn(
            "Refusing to run a QApplication with no topLevelWidgets. "
            f"To run the app anyway, use `{_func_name}(force=True)`"
        )
        return

    if app.thread().loopLevel() >= max_loop_level:
        loops = app.thread().loopLevel()
        s = 's' if loops > 1 else ''
        warn(
            f"A QApplication is already running with {loops} event loop{s}."
            "To enter *another* event loop, use "
            f"`{_func_name}(max_loop_level={loops + 1})`"
        )
        return

    with notification_manager:
        app.exec_()
