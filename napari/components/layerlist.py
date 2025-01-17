from __future__ import annotations

import itertools
import typing
import warnings
from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from napari.components.dims import RangeTuple
from napari.layers import Layer
from napari.layers.utils.layer_utils import Extent
from napari.utils.events.containers import SelectableEventedList
from napari.utils.naming import inc_name_count
from napari.utils.translations import trans

if TYPE_CHECKING:
    from npe2.manifest.io import WriterContribution
    from typing_extensions import Self


def get_name(layer: Layer) -> str:
    """Return the name of a layer."""
    return layer.name


class LayerList(SelectableEventedList[Layer]):
    """List-like layer collection with built-in reordering and callback hooks.

    Parameters
    ----------
    data : iterable
        Iterable of napari.layer.Layer

    Events
    ------
    inserting : (index: int)
        emitted before an item is inserted at ``index``
    inserted : (index: int, value: T)
        emitted after ``value`` is inserted at ``index``
    removing : (index: int)
        emitted before an item is removed at ``index``
    removed : (index: int, value: T)
        emitted after ``value`` is removed at ``index``
    moving : (index: int, new_index: int)
        emitted before an item is moved from ``index`` to ``new_index``
    moved : (index: int, new_index: int, value: T)
        emitted after ``value`` is moved from ``index`` to ``new_index``
    changed : (index: int, old_value: T, value: T)
        emitted when item at ``index`` is changed from ``old_value`` to ``value``
    changed <OVERLOAD> : (index: slice, old_value: List[_T], value: List[_T])
        emitted when item at ``index`` is changed from ``old_value`` to ``value``
    reordered : (value: self)
        emitted when the list is reordered (eg. moved/reversed).
    selection.events.changed : (added: Set[_T], removed: Set[_T])
        emitted when the set changes, includes item(s) that have been added
        and/or removed from the set.
    selection.events.active : (value: _T)
        emitted when the current item has changed.
    selection.events._current : (value: _T)
        emitted when the current item has changed. (Private event)

    Notes
    -----

    Note that ``changed`` events are only emitted when the value of an item in
    the layer list item value is changed, not when the item is inserted or removed.

    Examples
    --------

    >>> import napari
    >>> from skimage.data import astronaut
    >>> viewer = napari.Viewer()
    >>> event_list = []

    Connect to the event list:

    >>> viewer.layers.events.connect(event_list.append)
    <built-in method append of list object at 0x7fc225764780>

    >>> viewer.add_image(astronaut())
    <Image layer 'Image' at 0x7f0fe7fa9e90>
    >>> viewer.add_points()
    <Points layer 'Points' at 0x7f102f962350>
    >>> viewer.layers
    [<Image layer 'Image' at 0x7f0fe7fa9e90>, <Points layer 'Points' at 0x7f102f962350>]

    Inspecting the list of events, we see:

    >>> event_list[0].type
    'inserting'
    >>> viewer.layers.pop(1)
    <Points layer 'Points' at 0x7f102f962350>
    >>> event_list[-1].type
    'removed'

    """

    def __init__(self, data=()) -> None:
        super().__init__(
            data=data,
            basetype=Layer,
            lookup={str: get_name},
        )
        self._create_contexts()

    def _create_contexts(self):
        """Create contexts to manage enabled/visible action/menu states.

        Connects LayerList and Selection[Layer] to their context keys to allow
        actions and menu items (in the GUI) to be dynamically enabled/disabled
        and visible/hidden based on the state of layers in the list.
        """

        # TODO: figure out how to move this context creation bit.
        # Ideally, the app should be aware of the layerlist, but not vice versa.
        # This could probably be done by having the layerlist emit events that
        # the app connects to, then the `_ctx` object would live on the app,
        # (not here)
        from napari._app_model.context import create_context
        from napari._app_model.context._layerlist_context import (
            LayerListContextKeys,
            LayerListSelectionContextKeys,
        )

        self._ctx = create_context(self)
        if self._ctx is not None:  # happens during Viewer type creation
            self._ctx_keys = LayerListContextKeys(self._ctx)
            self.events.inserted.connect(self._ctx_keys.update)
            self.events.removed.connect(self._ctx_keys.update)

            self._selection_ctx_keys = LayerListSelectionContextKeys(self._ctx)
            self.selection.events.changed.connect(
                self._selection_ctx_keys.update
            )

    def _process_delete_item(self, item: Layer):
        super()._process_delete_item(item)
        item.events.extent.disconnect(self._clean_cache)
        item.events._extent_augmented.disconnect(self._clean_cache)
        self._clean_cache()

    def _clean_cache(self):
        cached_properties = (
            'extent',
            '_extent_world',
            '_extent_world_augmented',
            '_step_size',
        )
        [self.__dict__.pop(p, None) for p in cached_properties]

    def __newlike__(self, data):
        return LayerList(data)

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
        existing_layers = {x.name for x in self if x is not layer}
        for _ in range(len(self)):
            if name in existing_layers:
                name = inc_name_count(name)
        return name

    def _update_name(self, event):
        """Coerce name of the layer in `event.layer`."""
        layer = event.source
        layer.name = self._coerce_name(layer.name, layer)

    def _ensure_unique(self, values, allow=()):
        bad = set(self._list) - set(allow)
        values = tuple(values) if isinstance(values, Iterable) else (values,)
        for v in values:
            if v in bad:
                raise ValueError(
                    trans._(
                        "Layer '{v}' is already present in layer list",
                        deferred=True,
                        v=v,
                    )
                )
        return values

    @typing.overload
    def __getitem__(self, item: Union[int, str]) -> Layer: ...

    @typing.overload
    def __getitem__(self, item: slice) -> Self: ...

    def __getitem__(self, item):
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        old = self._list[key]
        if isinstance(key, slice):
            value = self._ensure_unique(value, old)
        elif isinstance(key, int):
            (value,) = self._ensure_unique((value,), (old,))
        super().__setitem__(key, value)

    def insert(self, index: int, value: Layer):
        """Insert ``value`` before index."""
        (value,) = self._ensure_unique((value,))
        new_layer = self._type_check(value)
        new_layer.name = self._coerce_name(new_layer.name)
        self._clean_cache()
        new_layer.events.extent.connect(self._clean_cache)
        new_layer.events._extent_augmented.connect(self._clean_cache)
        super().insert(index, new_layer)

    def remove_selected(self):
        """Remove selected layers from LayerList, but first unlink them."""
        if not self.selection:
            return
        self.unlink_layers(self.selection)
        super().remove_selected()

    def toggle_selected_visibility(self):
        """Toggle visibility of selected layers"""
        for layer in self.selection:
            layer.visible = not layer.visible

    @cached_property
    def _extent_world(self) -> np.ndarray:
        """Extent of layers in world coordinates.

        Default to 2D with (-0.5, 511.5) min/ max values if no data is present.
        Corresponds to pixels centered at [0, ..., 511].

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        return self._get_extent_world([layer.extent for layer in self])

    @cached_property
    def _extent_world_augmented(self) -> np.ndarray:
        """Extent of layers in world coordinates.

        Default to 2D with (-0.5, 511.5) min/ max values if no data is present.
        Corresponds to pixels centered at [0, ..., 511].

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        return self._get_extent_world(
            [layer._extent_augmented for layer in self],
            augmented=True,
        )

    def _get_min_and_max(self, mins_list, maxes_list):
        # Reverse dimensions since it is the last dimensions that are
        # displayed.
        mins_list = [mins[::-1] for mins in mins_list]
        maxes_list = [maxes[::-1] for maxes in maxes_list]

        with warnings.catch_warnings():
            # Taking the nanmin and nanmax of an axis of all nan
            # raises a warning and returns nan for that axis
            # as we have do an explicit nan_to_num below this
            # behaviour is acceptable and we can filter the
            # warning
            warnings.filterwarnings(
                'ignore',
                message=str(
                    trans._('All-NaN axis encountered', deferred=True)
                ),
            )
            min_v = np.nanmin(
                list(itertools.zip_longest(*mins_list, fillvalue=np.nan)),
                axis=1,
            )
            max_v = np.nanmax(
                list(itertools.zip_longest(*maxes_list, fillvalue=np.nan)),
                axis=1,
            )

        # 512 element default extent as documented in `_get_extent_world`
        min_v = np.nan_to_num(min_v, nan=-0.5)
        max_v = np.nan_to_num(max_v, nan=511.5)

        # switch back to original order
        return min_v[::-1], max_v[::-1]

    def _get_extent_world(self, layer_extent_list, augmented=False):
        """Extent of layers in world coordinates.

        Default to 2D image-like with (0, 511) min/ max values if no data is present.
        Corresponds to image with 512 pixels in each dimension.

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        if len(self) == 0:
            min_v = np.zeros(self.ndim)
            max_v = np.full(self.ndim, 511.0)
            # image-like augmented extent is actually expanded by 0.5
            if augmented:
                min_v -= 0.5
                max_v += 0.5
        else:
            extrema = [extent.world for extent in layer_extent_list]
            mins = [e[0] for e in extrema]
            maxs = [e[1] for e in extrema]
            min_v, max_v = self._get_min_and_max(mins, maxs)

        return np.vstack([min_v, max_v])

    @cached_property
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

    def _step_size_from_scales(self, scales):
        # Reverse order so last axes of scale with different ndim are aligned
        scales = [scale[::-1] for scale in scales]
        full_scales = list(
            np.array(list(itertools.zip_longest(*scales, fillvalue=np.nan)))
        )
        # restore original order
        return np.nanmin(full_scales, axis=1)[::-1]

    def _get_step_size(self, layer_extent_list):
        if len(self) == 0:
            return np.ones(self.ndim)

        scales = [extent.step for extent in layer_extent_list]
        return self._step_size_from_scales(scales)

    def get_extent(self, layers: Iterable[Layer]) -> Extent:
        """
        Return extent for a given layer list.

        Extent bounds are inclusive. This function is useful for calculating the extent
        of a subset of layers when preparing and updating some supplementary layers.
        For example see the cross Vectors layer in the `multiple_viewer_widget` example.

        Parameters
        ----------
        layers : list of Layer
            list of layers for which extent should be calculated

        Returns
        -------
        extent : Extent
            extent for selected layers
        """
        extent_list = [layer.extent for layer in layers]
        return Extent(
            data=None,
            world=self._get_extent_world(extent_list),
            step=self._get_step_size(extent_list),
        )

    @cached_property
    def extent(self) -> Extent:
        """
        Extent of layers in data and world coordinates.

        Extent bounds are inclusive. See Layer.extent for a detailed explanation
        of how extents are calculated.
        """
        return self.get_extent(list(self))

    @property
    def _ranges(self) -> tuple[RangeTuple, ...]:
        """Get ranges for Dims.range in world coordinates."""
        ext = self.extent
        return tuple(
            RangeTuple(*x) for x in zip(ext.world[0], ext.world[1], ext.step)
        )

    @property
    def ndim(self) -> int:
        """Maximum dimensionality of layers.

        Defaults to 2 if no data is present.

        Returns
        -------
        ndim : int
        """
        return max((layer.ndim for layer in self), default=2)

    def _link_layers(
        self,
        method: str,
        layers: Optional[Iterable[Union[str, Layer]]] = None,
        attributes: Iterable[str] = (),
    ):
        # adding this method here allows us to emit an event when
        # layers in this group are linked/unlinked.  Which is necessary
        # for updating context
        from napari.layers.utils import _link_layers

        if layers is not None:
            layers = [self[x] if isinstance(x, str) else x for x in layers]  # type: ignore
        else:
            layers = self
        getattr(_link_layers, method)(layers, attributes)
        self.selection.events.changed(added={}, removed={})

    def link_layers(
        self,
        layers: Optional[Iterable[Union[str, Layer]]] = None,
        attributes: Iterable[str] = (),
    ):
        """
        Links the selected layers.

        Once layers are linked, any action performed on one layer will be
        performed on all linked layers at the same time.
        """
        return self._link_layers('link_layers', layers, attributes)

    def unlink_layers(
        self,
        layers: Optional[Iterable[Union[str, Layer]]] = None,
        attributes: Iterable[str] = (),
    ):
        """Unlinks previously linked layers.

        Changes to one of the layer's properties no longer result in the same
        changes to the previously linked layers.
        """
        return self._link_layers('unlink_layers', layers, attributes)

    def save(
        self,
        path: str,
        *,
        selected: bool = False,
        plugin: Optional[str] = None,
        _writer: Optional[WriterContribution] = None,
    ) -> list[str]:
        """Save all or only selected layers to a path using writer plugins.

        If ``plugin`` is not provided and only one layer is targeted, then we
        directly call the corresponding``napari_write_<layer_type>`` hook (see
        :ref:`single layer writer hookspecs <write-single-layer-hookspecs>`)
        which will loop through implementations and stop when the first one
        returns a non-``None`` result. The order in which implementations are
        called can be changed with the Plugin sorter in the GUI or with the
        corresponding hook's
        :meth:`~napari.plugins._hook_callers._HookCaller.bring_to_front`
        method.

        If ``plugin`` is not provided and multiple layers are targeted,
        then we call
        :meth:`~napari.plugins.hook_specifications.napari_get_writer` which
        loops through plugins to find the first one that knows how to handle
        the combination of layers and is able to write the file. If no plugins
        offer :meth:`~napari.plugins.hook_specifications.napari_get_writer` for
        that combination of layers then the default
        :meth:`~napari.plugins.hook_specifications.napari_get_writer` will
        create a folder and call ``napari_write_<layer_type>`` for each layer
        using the ``Layer.name`` variable to modify the path such that the
        layers are written to unique files in the folder.

        If ``plugin`` is provided and a single layer is targeted, then we
        call the ``napari_write_<layer_type>`` for that plugin, and if it fails
        we error.

        If ``plugin`` is provided and multiple layers are targeted, then
        we call we call
        :meth:`~napari.plugins.hook_specifications.napari_get_writer` for
        that plugin, and if it doesn`t return a ``WriterFunction`` we error,
        otherwise we call it and if that fails if it we error.

        Parameters
        ----------
        path : str
            A filepath, directory, or URL to open.  Extensions may be used to
            specify output format (provided a plugin is available for the
            requested format).
        selected : bool
            Optional flag to only save selected layers. False by default.
        plugin : str, optional
            Name of the plugin to use for saving. If None then all plugins
            corresponding to appropriate hook specification will be looped
            through to find the first one that can save the data.
        _writer : WriterContribution, optional
            private: npe2 specific writer override.

        Returns
        -------
        list of str
            File paths of any files that were written.
        """
        from napari.plugins.io import save_layers

        layers = (
            [x for x in self if x in self.selection]
            if selected
            else list(self)
        )

        if selected:
            msg = trans._('No layers selected', deferred=True)
        else:
            msg = trans._('No layers to save', deferred=True)

        if not layers:
            warnings.warn(msg)
            return []

        return save_layers(path, layers, plugin=plugin, _writer=_writer)
