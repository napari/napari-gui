import sys
from contextlib import contextmanager
from os.path import dirname, join

from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QApplication, QSplashScreen

from ..utils import perf


def _create_application() -> QApplication:
    """Return a new QApplication.
    """
    if not perf.USE_PERFMON:
        return QApplication(sys.argv)  # Normal stock QApplication.

    # Special version that times Qt Events.
    from .qt_timing import QApplicationWithTiming

    return QApplicationWithTiming(sys.argv)


@contextmanager
def gui_qt(*, startup_logo=False):
    """Start a Qt event loop in which to run the application.

    Parameters
    ----------
    startup_logo : bool
        Show a splash screen with the napari logo during startup.

    Notes
    -----
    This context manager is not needed if running napari within an interactive
    IPython session. In this case, use the ``%gui qt`` magic command, or start
    IPython with the Qt GUI event loop enabled by default by using
    ``ipython --gui=qt``.
    """
    splash_widget = None
    app = QApplication.instance()
    if not app:
        # if this is the first time the Qt app is being instantiated, we set
        # the name, so that we know whether to raise_ in Window.show()
        app = _create_application()
        app.setApplicationName('napari')
        if startup_logo:
            logopath = join(dirname(__file__), '..', 'resources', 'logo.png')
            splash_widget = QSplashScreen(QPixmap(logopath).scaled(400, 400))
            splash_widget.show()
    yield app
    # if the application already existed before this function was called,
    # there's no need to start it again.  By avoiding unnecessary calls to
    # ``app.exec_``, we avoid blocking.
    if app.applicationName() == 'napari':
        if splash_widget and startup_logo:
            splash_widget.close()
        app.exec_()
