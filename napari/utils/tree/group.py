from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Iterable, List, TypeVar, Union

from napari.utils.events.containers._selectable_list import (
    SelectableNestableEventedList,
)
from napari.utils.tree.node import Node

if TYPE_CHECKING:
    from napari.utils.events.containers._nested_list import MaybeNestedIndex

NodeType = TypeVar("NodeType", bound=Node)


class Group(Node, SelectableNestableEventedList[NodeType]):
    """An object that can contain other objects in a composite Tree pattern.

    The ``Group`` (aka composite) is an element that has sub-elements:
    which may be ``Nodes`` or other ``Groups``.  By inheriting from
    :class:`NestableEventedList`, ``Groups`` have basic python list-like
    behavior and emit events when modified.  The main addition in this class
    is that when objects are added to a ``Group``, they are assigned a
    ``.parent`` attribute pointing to the group, which is removed upon
    deletion from the group.

    For additional background on the composite design pattern, see:
    https://refactoring.guru/design-patterns/composite

    Parameters
    ----------
    children : Iterable[Node], optional
        Items to initialize the Group, by default ().  All items must be
        instances of ``Node``.
    name : str, optional
        A name/id for this group, by default "Group"
    """

    def __init__(
        self,
        children: Iterable[NodeType] = (),
        name: str = "Group",
        basetype=Node,
    ) -> None:
        Node.__init__(self, name=name)
        SelectableNestableEventedList.__init__(
            self,
            data=children,
            basetype=basetype,
            lookup={str: lambda e: e.name},
        )

    def __newlike__(self, iterable: Iterable):
        # NOTE: TRICKY!
        # whenever we slice into a group with group[start:end],
        # the super().__newlike__() call is going to create a new object
        # of the same type (Group), and then populate it with items in iterable
        # ...
        # However, `Group.insert` changes the parent of each item as
        # it gets inserted.  (The implication is that no Node can live in
        # multiple groups at the same time). This means that simply slicing
        # into a group will actually reparent *all* items in that group
        # (even if the resulting slice goes unused...).
        #
        # So, we call new._list.extend here to avoid that reparenting.
        # Though this may have its own negative consequences for typing/events?
        new = type(self)()
        new._basetypes = self._basetypes
        new._lookup = self._lookup.copy()
        new._list.extend(iterable)
        return new

    def __getitem__(self, key) -> Union[NodeType, Group[NodeType]]:
        return super().__getitem__(key)

    def __delitem__(self, key: MaybeNestedIndex):
        """Remove item at ``key``, and unparent."""
        if isinstance(key, (int, tuple)):
            self[key].parent = None  # type: ignore
        else:
            for item in self[key]:
                item.parent = None
        super().__delitem__(key)

    @staticmethod
    def _reparent(item, parent):
        if isinstance(item, Group):
            for el in item:
                Group._reparent(el, item)
        item.parent = parent

    def _process_inserted_item(self, item):
        # when an item gets finally inserted, there is a chance that it is a `Group` object that
        # was created by __newlike__ (and therefore has no parents); this happens due to `_type_check`
        # creating a __newlike__ when objects are lists. We cannot override this behaviour, otherwise
        # every time element types are checked, things get reparented (at best a million events, at
        # worst everything breaks). To fix this, we recursively reparent children after insertion.
        Group._reparent(item, self)

    def is_group(self) -> bool:
        """Return True, indicating that this ``Node`` is a ``Group``."""
        return True

    def __contains__(self, other):
        """Return true if ``other`` appears anywhere under this group."""
        return any(item is other for item in self.traverse())

    def traverse(
        self, leaves_only=False, with_ancestors=False
    ) -> Generator[NodeType, None, None]:
        """Recursive all nodes and leaves of the Group tree."""
        obj = self.root() if with_ancestors else self
        if not leaves_only:
            yield obj
        for child in obj:
            yield from child.traverse(leaves_only)

    def _render(self) -> List[str]:
        """Recursively return list of strings that can render ascii tree."""
        lines = [self._node_name()]

        for n, child in enumerate(self):
            spacer, bul = (
                ("   ", "└──") if n == len(self) - 1 else ("  │", "├──")
            )
            child_tree = child._render()
            lines.append(f"  {bul}" + child_tree.pop(0))
            lines.extend([spacer + lay for lay in child_tree])

        return lines
