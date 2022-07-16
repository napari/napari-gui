from typing import Optional

from app_model.backends.qt import QModelMenu
from qtpy.QtCore import QObject


def build_qmodel_menu(
    menu_id: str, title: str = None, parent: Optional[QObject] = None
) -> QModelMenu:
    """Build a QModelMenu from the napari app model

    Parameters
    ----------
    menu_id : str
        ID of a menu registered with napari._app.app.menus
    title : str
        Title of the menu
    parent : QObject
        Parent of the menu

    Returns
    -------
    QModelMenu
        QMenu subclass populated with all items in `menu_id` menu.
    """
    return QModelMenu(
        menu_id=menu_id, app='napari', title=title, parent=parent
    )
