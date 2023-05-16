from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING, Iterable, List, Optional

import numpy as np

from napari.components.layerlist import Extent, _LayerListMixin
from napari.layers.base import Layer
from napari.layers.utils.layer_utils import combine_extents
from napari.utils.naming import inc_name_count
from napari.utils.translations import trans
from napari.utils.tree import Group

if TYPE_CHECKING:
    from npe2.manifest.contributions import WriterContribution


class LayerGroup(Group[Layer], Layer, _LayerListMixin):
    def __init__(
        self, children: Iterable[Layer] = (), name: str = 'LayerGroup'
    ) -> None:
        Group.__init__(self, children, name=name, basetype=Layer)
        Layer.__init__(self, None, self._get_ndim(), name=name)

        # avoid circular import
        from napari._app_model.context import create_context
        from napari._app_model.context._layerlist_context import (
            LayerListContextKeys,
        )

        self.refresh(None)  # TODO: why...
        self.events.connect(self._handle_child_events)
        self._ctx = create_context(self)
        if self._ctx is not None:  # happens during Viewer type creation
            self._ctx_keys = LayerListContextKeys(self._ctx)

            self.selection.events.changed.connect(self._ctx_keys.update)
        # temporary: see note in _on_selection_event
        self.selection.events.changed.connect(self._on_selection_changed)

    def add_group(self, index=-1):
        lg = LayerGroup()
        self.insert(index, lg)
        return lg

    def _handle_child_events(self, event):
        # event.sources[0] is the original event emitter.
        # we only want child events here
        if not hasattr(event, 'index') or event.sources[0] is self:
            return
        if event.index != () and event.type == 'thumbnail':
            self._update_thumbnail()

    def __str__(self):
        return Group.__str__(self)

    def __repr__(self):
        return Group.__repr__(self)

    def _coerce_name(self, name, layer=None):
        """Coerce a name into a unique equivalent.

        Parameters
        ----------
        name : str
            Original name.
        layer : napari.layers.Layer, optional
            Layer for which name is generated.

        Returns
        -------
        new_name : str
            Coerced, unique name.
        """

        for existing_name in sorted(
            x.name
            for x in self.traverse(with_ancestors=True)
            if x is not layer
        ):
            if name == existing_name:
                name = inc_name_count(name)
        return name

    def _update_name(self, event):
        """Coerce name of the layer in `event.layer`."""
        layer = event.source
        layer.name = self._coerce_name(layer.name, layer)

    def _update_ndim(self):
        ndim = self._get_ndim()
        if ndim == self._ndim:
            return
        self._construct_transform_chain(
            ndim=ndim,
            scale=None,
            translate=None,
            rotate=None,
            shear=None,
            affine=None,
        )

    @property
    def _transforms(self):
        self._update_ndim()
        return self.__transforms

    @_transforms.setter
    def _transforms(self, value):
        self.__transforms = value

    def insert(self, index: int, value: Layer):
        """Insert ``value`` before index."""
        # temporarily disabled while we work on nested selection bug and
        # group vispy nodes.
        # once removed, uncomment "add group" button in qt_viewer_buttons
        if isinstance(value, (LayerGroup, list)):
            import os

            if not os.getenv("ALLOW_LAYERGROUPS"):
                warnings.warn(
                    "Nested layergroups not quite ready. "
                    "Enabled with env var ALLOW_LAYERGROUPS=1."
                )
                return

        new_layer = self._type_check(value)
        new_layer.name = self._coerce_name(new_layer.name)
        super().insert(index, new_layer)
        self._update_thumbnail()

    def __delitem__(self, key):
        """Remove item at `key`."""
        super().__delitem__(key)
        self._update_thumbnail()

    @property
    def _extent_data(self):
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        return combine_extents([c._extent_data for c in self])

    @property
    def _extent_world(self) -> np.ndarray:
        """Extent of layers in world coordinates.

        Default to 2D with (0, 512) min/ max values if no data is present.

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        return self._get_extent_world([layer.extent for layer in self])

    def _get_extent_world(self, layer_extent_list):
        """Extent of layers in world coordinates.

        Default to 2D with (-0.5, 511.5) min/ max values if no data is present.
        Corresponds to pixels centered at [0, ..., 511].

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        if len(self) == 0:
            min_v = np.asarray([-0.5] * self.ndim)
            max_v = np.asarray([511.5] * self.ndim)
        else:
            extrema = [extent.world for extent in layer_extent_list]
            mins = [e[0] for e in extrema]
            maxs = [e[1] for e in extrema]
            min_v, max_v = self._get_min_and_max(mins, maxs)

        return np.vstack([min_v, max_v])

    @property
    def _step_size(self) -> np.ndarray:
        """Ideal step size between planes in world coordinates.

        Computes the best step size that allows all data planes to be
        sampled if moving through the full range of world coordinates.
        The current implementation just takes the minimum scale.

        Returns
        -------
        step_size : array, shape (D,)
        """
        return self._get_step_size([layer.extent for layer in self])

    def _get_step_size(self, layer_extent_list):
        if len(self) == 0:
            return np.ones(self.ndim)

        scales = [extent.step[::-1] for extent in layer_extent_list]
        full_scales = list(
            np.array(list(itertools.zip_longest(*scales, fillvalue=np.nan))).T
        )
        min_scales = np.nanmin(full_scales, axis=0)
        return min_scales[::-1]

    @property
    def extent(self) -> Extent:
        """Extent of layers in data and world coordinates."""
        extent_list = [layer.extent for layer in self]
        return Extent(
            data=None,
            world=self._get_extent_world(extent_list),
            step=self._get_step_size(extent_list),
        )

    @property
    def ndim(self) -> int:
        return self._get_ndim()

    def _get_ndim(self):
        return max((c._get_ndim() for c in self), default=2)

    def _get_state(self):
        """LayerGroup state as a list of state dictionaries.

        Returns
        -------
        state : list
            List of layer state dictionaries.
        """
        state = [self._get_base_state()]
        if self is not None:
            state.extend(layer._get_state() for layer in self)
        return state

    def _get_value(self, position):
        """Returns a flat list of all layer values in the layergroup
        for a given mouse position and set of indices.

        Layers in layergroup are iterated over by depth-first recursive search.

        Returns
        ----------
        value : list
            Flat list containing values of the layer data at the coord.
        """
        return [layer._get_value(position) for layer in self]

    def _set_view_slice(self):
        """Set the view for each layer given the indices to slice with."""
        for child in self:
            child._set_view_slice()

    def _set_highlight(self, force=False):
        """Render layer hightlights when appropriate."""
        for child in self:
            child._set_highlight(force=force)

    def _update_thumbnail(self, *args, **kwargs):
        if not hasattr(self, '_thumbnail_shape'):
            # layer is not finished initializing
            return
        leaves = list(self.traverse(leaves_only=True))
        if len(leaves) > 1:
            # TODO: this doesn't take blending into account...
            thumb = np.clip(
                np.sum([leaf.thumbnail for leaf in leaves], axis=0), 0, 255
            )
        elif leaves:
            thumb = leaves[0].thumbnail
        else:
            thumb = np.zeros(self._thumbnail_shape)
            thumb[..., 3] = 255
        self.thumbnail = thumb

    def refresh(self, event=None):
        """Refresh all layer data if visible."""
        if self.visible:
            for child in self:
                child.refresh()

    @property
    def data(self):
        return None

    def _update_draw(self, *a, **k):
        return

    def save(
        self,
        path: str,
        *,
        selected: bool = False,
        plugin: Optional[str] = None,
        _writer: Optional[WriterContribution] = None,
    ) -> List[str]:
        from napari.plugins.io import save_layers

        layers = (
            list(self.selection)
            if selected
            else list(self.traverse(leaves_only=True))
        )

        if selected:
            msg = trans._("No layers selected", deferred=True)
        else:
            msg = trans._("No layers to save", deferred=True)

        if not layers:
            warnings.warn(msg)
            return []

        return save_layers(path, layers, plugin=plugin, _writer=_writer)
