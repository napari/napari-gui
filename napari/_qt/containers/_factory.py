from __future__ import annotations

from typing import TYPE_CHECKING, Union

from napari.components.layerlist import LayerList
from napari.layers.layergroup import LayerGroup
from napari.utils.events import SelectableEventedList
from napari.utils.translations import trans
from napari.utils.tree import Group

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


def create_view(
    obj: Union[SelectableEventedList, Group], parent: QWidget = None
):
    """Create a `QtListView`, or `QtNodeTreeView` for `obj`.

    Parameters
    ----------
    obj : SelectableEventedList or Group
        The python object for which to creat a QtView.
    parent : QWidget, optional
        Optional parent widget, by default None

    Returns
    -------
    Union[QtListView, QtNodeTreeView]
        A view instance appropriate for `obj`.
    """
    from napari._qt.containers import QtLayerList, QtListView, QtNodeTreeView

    if isinstance(obj, LayerList):
        return QtLayerList(obj, parent=parent)
    if isinstance(obj, Group):
        return QtNodeTreeView(obj, parent=parent)
    if isinstance(obj, SelectableEventedList):
        return QtListView(obj, parent=parent)
    raise TypeError(
        trans._(
            "Cannot create Qt view for obj: {obj}",
            deferred=True,
            obj=obj,
        )
    )


def create_model(
    obj: Union[SelectableEventedList, Group], parent: QWidget = None
):
    """Create a `QtListModel`, or `QtNodeTreeModel` for `obj`.

    Parameters
    ----------
    obj : SelectableEventedList or Group
        The python object for which to creat a QtView.
    parent : QWidget, optional
        Optional parent widget, by default None

    Returns
    -------
    Union[QtListModel, QtNodeTreeModel]
        A model instance appropriate for `obj`.
    """
    from napari._qt.containers import (
        QtLayerListModel,
        QtLayerTreeModel,
        QtListModel,
        QtNodeTreeModel,
    )

    if isinstance(obj, LayerList):
        return QtLayerListModel(obj, parent=parent)
    if isinstance(obj, LayerGroup):
        return QtLayerTreeModel(obj, parent=parent)
    if isinstance(obj, Group):
        return QtNodeTreeModel(obj, parent=parent)
    if isinstance(obj, SelectableEventedList):
        return QtListModel(obj, parent=parent)
    raise TypeError(
        trans._(
            "Cannot create Qt model for obj: {obj}",
            deferred=True,
            obj=obj,
        )
    )
