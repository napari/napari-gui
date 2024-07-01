import typing
from typing import TYPE_CHECKING, Optional
from weakref import WeakSet

import magicgui as mgui

from napari.components.viewer_model import ViewerModel
from napari.utils import _magicgui

if TYPE_CHECKING:
    # helpful for IDE support
    from napari._qt.qt_main_window import Window


@mgui.register_type(bind=_magicgui.proxy_viewer_ancestor)
class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    title : string, optional
        The title of the viewer window. By default 'napari'.
    ndisplay : {2, 3}, optional
        Number of displayed dimensions. By default 2.
    order : tuple of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3. By default None
    axis_labels : list of str, optional
        Dimension names. By default they are labeled with sequential numbers
    show : bool, optional
        Whether to show the viewer after instantiation. By default True.
    """

    _window: 'Window' = None  # type: ignore
    _instances: typing.ClassVar[WeakSet['Viewer']] = WeakSet()

    def __init__(
        self,
        *,
        title='napari',
        ndisplay=2,
        order=(),
        axis_labels=(),
        show=True,
        **kwargs,
    ) -> None:
        super().__init__(
            title=title,
            ndisplay=ndisplay,
            order=order,
            axis_labels=axis_labels,
            **kwargs,
        )
        # we delay initialization of plugin system to the first instantiation
        # of a viewer... rather than just on import of plugins module
        from napari.plugins import _initialize_plugins

        # having this import here makes all of Qt imported lazily, upon
        # instantiating the first Viewer.
        from napari.window import Window

        _initialize_plugins()

        self._window = Window(self, show=show)
        self._instances.add(self)

    # Expose private window publically. This is needed to keep window off pydantic model
    @property
    def window(self) -> 'Window':
        return self._window

    def update_console(self, variables):
        """Update console's namespace with desired variables.

        Parameters
        ----------
        variables : dict, str or list/tuple of str
            The variables to inject into the console's namespace.  If a dict, a
            simple update is done.  If a str, the string is assumed to have
            variable names separated by spaces.  A list/tuple of str can also
            be used to give the variable names.  If just the variable names are
            give (list/tuple/str) then the variable values looked up in the
            callers frame.
        """
        if self.window._qt_viewer._console is None:
            self.window._qt_viewer.add_to_console_backlog(variables)
            return
        self.window._qt_viewer.console.push(variables)

    def export_figure(
        self,
        path: Optional[str] = None,
        *,
        scale: float = 1,
        flash: bool = True,
    ):
        """Export an image of the full extent of the displayed layer data.

        This function finds a tight boundary around the data, resets the view
        around that boundary (and, when scale=1, such that 1 captured pixel is
        equivalent to one data pixel), takes a screenshot, then restores the
        previous zoom and canvas sizes. For example with one image layer with
        an image of 256 x 256 pixels and layer.scale=[1,1], with scale=1 the
        size of the resulting figure will be 256 x 256 pixels. If layer.scale
        for both dimensions is set to 0.5, it will be 128 x 128 pixels. Currently,
        only works when 2 dimensions are displayed.

        Parameters
        ----------
        path : Optional[str]
            Filename for saving screenshot image.
        scale : float
            Scale factor used to increase resolution of canvas for the
            screenshot. By default, a scale of 1. A scale
            of 1 corresponds to 1 data pixel per screenshot pixel if all displayed
            layers have layer.scale for each displayed dimension set to 1.
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
        return self.window.export_figure(
            path=path,
            scale=scale,
            flash=flash,
        )

    def screenshot(
        self,
        path: Optional[str] = None,
        *,
        size: Optional[tuple[str, str]] = None,
        scale: Optional[float] = None,
        canvas_only: bool = True,
        flash: bool = True,
    ):
        """Take currently displayed screen and convert to an image array.

        Parameters
        ----------
        path : str, optional
            Filename for saving screenshot image.
        size : tuple of two ints, optional
            Size (resolution height x width) of the screenshot. By default, the currently displayed size.
            Only used if `canvas_only` is True.
        scale : float, optional
            Scale factor used to increase resolution of canvas for the screenshot. By default, the currently displayed resolution.
            Only used if `canvas_only` is True.
        canvas_only : bool
            If True, screenshot shows only the image display canvas, and
            if False include the napari viewer frame in the screenshot,
            By default, True.
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
        return self.window.screenshot(
            path=path,
            size=size,
            scale=scale,
            flash=flash,
            canvas_only=canvas_only,
        )

    def show(self, *, block=False):
        """Resize, show, and raise the viewer window."""
        self.window.show(block=block)

    def close(self):
        """Close the viewer window."""
        # Shutdown the slicer first to avoid processing any more tasks.
        self._layer_slicer.shutdown()
        # Remove all the layers from the viewer
        self.layers.clear()
        # Close the main window
        self.window.close()

        self._instances.discard(self)

    @classmethod
    def close_all(cls) -> int:
        """
        Class metod, Close all existing viewer instances.

        This is mostly exposed to avoid leaking of viewers when running tests.
        As having many non-closed viewer can adversely affect performances.

        It will return the number of viewer closed.

        Returns
        -------
        int
            number of viewer closed.

        """
        # copy to not iterate while changing.
        viewers = list(cls._instances)
        ret = len(viewers)
        for viewer in viewers:
            viewer.close()
        return ret


def current_viewer() -> Optional[Viewer]:
    """Return the currently active napari viewer."""
    try:
        from napari._qt.qt_main_window import _QtMainWindow
    except ImportError:
        return None
    else:
        return _QtMainWindow.current_viewer()
