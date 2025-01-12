from collections.abc import Generator
from typing import TYPE_CHECKING

from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.utils.tree.group import Group


class Node:
    """An object that can be a member of a :class:`Group`.

    ``Node`` forms the base object of a composite Tree pattern. This class
    describes operations that are common to both simple (node) and complex
    (group) elements of the tree.  ``Node`` may not have children, whereas
    :class:`~napari.utils.tree.group.Group` can.

    For additional background on the composite design pattern, see:
    https://refactoring.guru/design-patterns/composite

    Parameters
    ----------
    name : str, optional
        A name/id for this node, by default "Node"

    Attributes
    ----------
    parent : Group, optional
        The parent of this Node.
    """

    def __init__(self, name: str = 'Node') -> None:
        self.parent: Group | None = None
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def is_group(self) -> bool:
        """Return True if this Node is a composite.

        :class:`~napari.utils.tree.Group` will return True.
        """
        return False

    def index_in_parent(self) -> int | None:
        """Return index of this Node in its parent, or None if no parent."""
        return self.parent.index(self) if self.parent is not None else None

    def index_from_root(self) -> tuple[int, ...]:
        """Return index of this Node relative to root.

        Will return ``()`` if this object *is* the root.
        """
        item = self
        indices: list[int] = []
        while item.parent is not None:
            indices.insert(0, item.index_in_parent())  # type: ignore
            item = item.parent
        return tuple(indices)

    def iter_parents(self):
        """Iterate the parent chain, starting with nearest relatives"""
        obj = self.parent
        while obj:
            yield obj
            obj = obj.parent

    def root(self) -> 'Node':
        """Get the root parent."""
        parents = list(self.iter_parents())
        return parents[-1] if parents else self

    def traverse(
        self, leaves_only=False, with_ancestors=False
    ) -> Generator['Node', None, None]:
        """Recursive all nodes and leaves of the Node.

        This is mostly used by :class:`~napari.utils.tree.Group`, which can
        also traverse children.  A ``Node`` simply yields itself.
        """
        yield self

    def __str__(self):
        """Render ascii tree string representation of this node"""
        return '\n'.join(self._render())

    def _render(self) -> list[str]:
        """Return list of strings that can render ascii tree.

        For ``Node``, we just return the name of this specific node.
        :class:`~napari.utils.tree.Group` will render a full tree.
        """
        return [self._node_name()]

    def _node_name(self) -> str:
        """Will be used when rendering node tree as string.

        Subclasses may override as desired.
        """
        return self.name

    def unparent(self):
        """Remove this object from its parent."""
        if self.parent is not None:
            self.parent.remove(self)
            return self
        raise IndexError(
            trans._(
                'Cannot unparent orphaned Node: {node!r}',
                deferred=True,
                node=self,
            ),
        )
