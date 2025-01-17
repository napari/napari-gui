from typing import Union

from app_model.expressions import parse_expression
from app_model.types import Action, MenuItem, SubmenuItem

from napari._app_model import get_app_model
from napari._app_model.constants import MenuGroup, MenuId

MenuOrSubmenu = Union[MenuItem, SubmenuItem]


def to_id_key(menu_path: str) -> str:
    """Return final part of the menu path.

    Parameters
    ----------
    menu_path : str
        full string delineating the menu path

    Returns
    -------
    str
        final part of the menu path
    """
    return menu_path.split('/')[-1]


def to_action_id(id_key: str) -> str:
    """Return dummy action ID for the given id_key.

    Parameters
    ----------
    id_key : str
        key to use in action ID

    Returns
    -------
    str
        dummy action ID
    """
    return f'napari.{id_key}.empty_dummy'


def contains_dummy_action(menu_items: list[MenuOrSubmenu]) -> bool:
    """True if one of the menu_items is the dummy action, otherwise False.

    Parameters
    ----------
    menu_items : list[MenuOrSubmenu]
        menu items belonging to a given menu

    Returns
    -------
    bool
        True if menu_items contains dummy item otherwise false
    """
    for item in menu_items:
        if hasattr(item, 'command') and 'empty_dummy' in item.command.id:
            return True
    return False


def is_empty_menu(menu_id: str) -> bool:
    """Return True if the given menu_id is empty, otherwise False

    Parameters
    ----------
    menu_id : str
        id of the menu to check

    Returns
    -------
    bool
        True if the given menu_id is empty, otherwise False
    """
    app = get_app_model()
    if menu_id not in app.menus:
        return True
    if len(app.menus.get_menu(menu_id)) == 0:
        return True
    return bool(
        len(app.menus.get_menu(menu_id)) == 1
        and contains_dummy_action(app.menus.get_menu(menu_id))
    )


def no_op() -> None:
    """Fully qualified no-op to use for dummy actions."""


def get_dummy_action(menu_id: MenuId) -> tuple[Action, str]:
    """Return a dummy action to be used for the given menu.

    The part of the menu_id after the final `/` will form
    a unique id_key used for the action ID and the when
    expression context key.

    Parameters
    ----------
    menu_id: MenuId
        id of the menu to add the dummy action to

    Returns
    -------
    tuple[Action, str]
        dummy action and the `when` expression context key
    """
    # NOTE: this assumes the final word of each contributable
    # menu path is unique, otherwise, we will clash. Once we
    # move to using short menu keys, the key itself will be used
    # here and this will no longer be a concern.
    id_key = to_id_key(menu_id)
    action = Action(
        id=to_action_id(id_key),
        title='Empty',
        callback=no_op,
        menus=[
            {
                'id': menu_id,
                'group': MenuGroup.NAVIGATION,
                'when': parse_expression(context_key := f'{id_key}_empty'),
            }
        ],
        enablement=False,
        palette=False,
    )
    return action, context_key
