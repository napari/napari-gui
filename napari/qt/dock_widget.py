"""
Currently just passes some dock_widget methods to the main window.
"""
from typing import TYPE_CHECKING, Optional, Sequence, Union

from qtpy.QtWidgets import QMenu, QWidget

from napari.viewer import current_viewer

if TYPE_CHECKING:
    from magicgui.widgets import Widget


class DockWidget:
    @staticmethod
    def add_dock_widget(
        widget: Union[QWidget, 'Widget'],
        *,
        name: str = '',
        area: str = 'right',
        allowed_areas: Optional[Sequence[str]] = None,
        add_vertical_stretch=True,
        tabify: bool = False,
        menu: Optional[QMenu] = None,
    ):
        """Add a widget to the napari dock area.

        Parameters
        ----------
        widget : QWidget or magicgui.Widget
            The widget to add to the dock area.
        name : str, optional
            The name of the widget, by default ''
        area : str, optional
            The area to add the widget to, by default 'right'
        allowed_areas : Sequence[str], optional
            A list of areas where the widget can be moved, by default None
        add_vertical_stretch : bool, optional
            Whether to add a vertical stretch to the widget, by default True
        tabify : bool, optional
            Whether to tabify the widget with any existing widgets, by default False
        menu : QMenu, optional
            A menu to add the widget to, by default None
        """
        current_viewer().window.add_dock_widget(
            widget,
            name=name,
            area=area,
            allowed_areas=allowed_areas,
            add_vertical_stretch=add_vertical_stretch,
            tabify=tabify,
            menu=menu,
        )

    @staticmethod
    def add_function_widget(
        function,
        *,
        magic_kwargs=None,
        name: str = '',
        area=None,
        allowed_areas=None,
        shortcut=None,
    ):
        """Turn a function into a dock widget via magicgui and adds it to the napari dock area.

        Parameters
        ----------
        function : callable
            The function to add to the dock area.
        magic_kwargs : dict, optional
            Keyword arguments to pass to magicgui, by default None
        name : str, optional
            The name of the widget, by default ''
        area : str, optional
            The area to add the widget to, by default None
        allowed_areas : list[str], optional
            A list of areas where the widget can be moved, by default None
        shortcut : str, optional
            A keyboard shortcut to call the function, by default None
        """
        current_viewer().window.add_function_widget(
            function,
            magic_kwargs=magic_kwargs,
            name=name,
            area=area,
            allowed_areas=allowed_areas,
            shortcut=shortcut,
        )

    @staticmethod
    def remove_dock_widget(widget: QWidget, menu=None):
        """Remove a widget from the napari dock area.

        Parameters
        ----------
        widget : QWidget | str
            The widget to remove from the dock area. If widget == 'all', all docked widgets will be removed.
        menu : QMenu, optional
            A menu to remove the widget from, by default None
        """
        current_viewer().window.remove_dock_widget(widget, menu=menu)

    @staticmethod
    def get_dock_widgets():
        """A dict of all dock widgets."""
        return current_viewer().window._dock_widgets
