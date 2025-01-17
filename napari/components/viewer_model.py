from __future__ import annotations

import inspect
import itertools
import os
import warnings
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
    cast,
)

import numpy as np

# This cannot be condition to TYPE_CHECKING or the stubgen fails
# with undefined Context.
from app_model.expressions import Context

from napari import layers
from napari._pydantic_compat import Extra, Field, PrivateAttr, validator
from napari.components._layer_slicer import _LayerSlicer
from napari.components._viewer_mouse_bindings import (
    dims_scroll,
    double_click_to_zoom,
)
from napari.components.camera import Camera
from napari.components.cursor import Cursor, CursorStyle
from napari.components.dims import Dims
from napari.components.grid import GridCanvas
from napari.components.layerlist import LayerList
from napari.components.overlays import (
    AxesOverlay,
    BrushCircleOverlay,
    Overlay,
    ScaleBarOverlay,
    TextOverlay,
)
from napari.components.tooltip import Tooltip
from napari.errors import (
    MultipleReaderError,
    NoAvailableReaderError,
    ReaderPluginError,
)
from napari.layers import (
    Image,
    Labels,
    Layer,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from napari.layers._source import Source, layer_source
from napari.layers.image._image_key_bindings import image_fun_to_mode
from napari.layers.image._image_utils import guess_labels
from napari.layers.labels._labels_key_bindings import labels_fun_to_mode
from napari.layers.points._points_key_bindings import points_fun_to_mode
from napari.layers.shapes._shapes_key_bindings import shapes_fun_to_mode
from napari.layers.surface._surface_key_bindings import surface_fun_to_mode
from napari.layers.tracks._tracks_key_bindings import tracks_fun_to_mode
from napari.layers.utils.stack_utils import split_channels
from napari.layers.vectors._vectors_key_bindings import vectors_fun_to_mode
from napari.plugins.utils import get_potential_readers, get_preferred_reader
from napari.settings import get_settings
from napari.types import (
    FullLayerData,
    LayerData,
    LayerTypeName,
    PathLike,
    PathOrPaths,
    SampleData,
)
from napari.utils._register import create_func as create_add_method
from napari.utils.action_manager import action_manager
from napari.utils.colormaps import ensure_colormap
from napari.utils.events import (
    Event,
    EventedDict,
    EventedModel,
    disconnect_events,
)
from napari.utils.events.event import WarningEmitter
from napari.utils.key_bindings import KeymapProvider
from napari.utils.migrations import rename_argument
from napari.utils.misc import is_sequence
from napari.utils.mouse_bindings import MousemapProvider
from napari.utils.progress import progress
from napari.utils.theme import available_themes, is_theme_available
from napari.utils.translations import trans

if TYPE_CHECKING:
    from npe2.types import SampleDataCreator


DEFAULT_THEME = 'dark'
EXCLUDE_DICT = {
    'keymap',
    '_mouse_wheel_gen',
    '_mouse_drag_gen',
    '_persisted_mouse_event',
    'mouse_move_callbacks',
    'mouse_drag_callbacks',
    'mouse_wheel_callbacks',
}
EXCLUDE_JSON = EXCLUDE_DICT.union({'layers', 'active_layer'})
Dict = dict  # rename, because ViewerModel has method dict

__all__ = ['ViewerModel', 'valid_add_kwargs']


def _current_theme() -> str:
    return get_settings().appearance.theme


DEFAULT_OVERLAYS = {
    'scale_bar': ScaleBarOverlay,
    'text': TextOverlay,
    'axes': AxesOverlay,
    'brush_circle': BrushCircleOverlay,
}


# KeymapProvider & MousemapProvider should eventually be moved off the ViewerModel
class ViewerModel(KeymapProvider, MousemapProvider, EventedModel):
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders, and control bars for color limits.

    Parameters
    ----------
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str
        Dimension names.

    Attributes
    ----------
    camera: napari.components.camera.Camera
        The camera object modeling the position and view.
    cursor: napari.components.cursor.Cursor
        The cursor object containing the position and properties of the cursor.
    dims : napari.components.dims.Dimensions
        Contains axes, indices, dimensions and sliders.
    grid: napari.components.grid.Gridcanvas
        Gridcanvas allowing for the current implementation of a gridview of the canvas.
    help: str
        A help message of the viewer model
    layers : napari.components.layerlist.LayerList
        List of contained layers.
    mouse_over_canvas: bool
        Indicating whether the mouse cursor is on the viewer canvas.
    theme: str
        Name of the Napari theme of the viewer
    title: str
        The title of the viewer model
    tooltip: napari.components.tooltip.Tooltip
        A tooltip showing extra information on the cursor
    window : napari._qt.qt_main_window.Window
        Parent window.
    _canvas_size: Tuple[int, int]
        The canvas size following the Numpy convention of height x width
    _ctx: Mapping
        Viewer object context mapping.
    _layer_slicer: napari.components._layer_slicer._Layer_Slicer
        A layer slicer object controlling the creation of a slice
    _overlays: napari.utils.events.containers._evented_dict.EventedDict[str, Overlay]
        An EventedDict with as keys the string names of different napari overlays and as values the napari.Overlay
        objects.
    """

    # Using allow_mutation=False means these attributes aren't settable and don't
    # have an event emitter associated with them
    camera: Camera = Field(default_factory=Camera, allow_mutation=False)
    cursor: Cursor = Field(default_factory=Cursor, allow_mutation=False)
    dims: Dims = Field(default_factory=Dims, allow_mutation=False)
    grid: GridCanvas = Field(default_factory=GridCanvas, allow_mutation=False)
    layers: LayerList = Field(
        default_factory=LayerList, allow_mutation=False
    )  # Need to create custom JSON encoder for layer!
    help: str = ''
    status: Union[str, dict] = 'Ready'
    tooltip: Tooltip = Field(default_factory=Tooltip, allow_mutation=False)
    theme: str = Field(default_factory=_current_theme)
    title: str = 'napari'
    # private track of overlays, only expose the old ones for backward compatibility
    _overlays: EventedDict[str, Overlay] = PrivateAttr(
        default_factory=EventedDict
    )
    # 2-tuple indicating height and width
    _canvas_size: tuple[int, int] = (800, 600)
    _ctx: Context
    # To check if mouse is over canvas to avoid race conditions between
    # different events systems
    mouse_over_canvas: bool = False

    # Need to use default factory because slicer is not copyable which
    # is required for default values.
    _layer_slicer: _LayerSlicer = PrivateAttr(default_factory=_LayerSlicer)

    def __init__(
        self, title='napari', ndisplay=2, order=(), axis_labels=()
    ) -> None:
        # max_depth=0 means don't look for parent contexts.
        from napari._app_model.context import create_context

        # FIXME: just like the LayerList, this object should ideally be created
        # elsewhere.  The app should know about the ViewerModel, but not vice versa.
        self._ctx = create_context(self, max_depth=0)
        # allow extra attributes during model initialization, useful for mixins
        self.__config__.extra = Extra.allow
        super().__init__(
            title=title,
            dims={
                'axis_labels': axis_labels,
                'ndisplay': ndisplay,
                'order': order,
            },
        )
        self.__config__.extra = Extra.ignore

        settings = get_settings()
        self.tooltip.visible = settings.appearance.layer_tooltip_visibility
        settings.appearance.events.layer_tooltip_visibility.connect(
            self._tooltip_visible_update
        )

        self._update_viewer_grid()
        settings.application.events.grid_stride.connect(
            self._update_viewer_grid
        )
        settings.application.events.grid_width.connect(
            self._update_viewer_grid
        )
        settings.application.events.grid_height.connect(
            self._update_viewer_grid
        )
        settings.experimental.events.async_.connect(self._update_async)

        # Add extra events - ideally these will be removed too!
        self.events.add(
            layers_change=WarningEmitter(
                trans._(
                    'This event will be removed in 0.5.0. Please use viewer.layers.events instead',
                    deferred=True,
                ),
                type_name='layers_change',
            ),
            reset_view=Event,
        )

        # Connect events
        self.grid.events.connect(self.reset_view)
        self.grid.events.connect(self._on_grid_change)
        self.dims.events.ndisplay.connect(self._update_layers)
        self.dims.events.ndisplay.connect(self.reset_view)
        self.dims.events.order.connect(self._update_layers)
        self.dims.events.order.connect(self.reset_view)
        self.dims.events.point.connect(self._update_layers)
        # FIXME: the next line is a temporary workaround. With #5522 and #5751 Dims.point became
        #        the source of truth, and is now defined in world space. This exposed an existing
        #        bug where if a field in Dims is modified by the root_validator, events won't
        #        be fired for it. This won't happen for properties because we have dependency
        #        checks. To fix this, we need dep checks for fields (psygnal!) and then we
        #        can remove the following line. Note that because of this we fire double events,
        #        but this should be ok because we have early returns when slices are unchanged.
        self.dims.events.current_step.connect(self._update_layers)
        self.dims.events.margin_left.connect(self._update_layers)
        self.dims.events.margin_right.connect(self._update_layers)
        self.cursor.events.position.connect(self.update_status_from_cursor)
        self.layers.events.inserted.connect(self._on_add_layer)
        self.layers.events.removed.connect(self._on_remove_layer)
        self.layers.events.reordered.connect(self._on_grid_change)
        self.layers.events.reordered.connect(self._on_layers_change)
        self.layers.selection.events.active.connect(self._on_active_layer)

        # Add mouse callback
        self.mouse_wheel_callbacks.append(dims_scroll)
        self.mouse_double_click_callbacks.append(double_click_to_zoom)

        self._overlays.update({k: v() for k, v in DEFAULT_OVERLAYS.items()})

    # simple properties exposing overlays for backward compatibility
    @property
    def axes(self):
        return self._overlays['axes']

    @property
    def scale_bar(self):
        return self._overlays['scale_bar']

    @property
    def text_overlay(self):
        return self._overlays['text']

    @property
    def _brush_circle_overlay(self):
        return self._overlays['brush_circle']

    def _tooltip_visible_update(self, event):
        self.tooltip.visible = event.value

    def _update_viewer_grid(self):
        """Keep viewer grid settings up to date with settings values."""

        settings = get_settings()

        self.grid.stride = settings.application.grid_stride
        self.grid.shape = (
            settings.application.grid_height,
            settings.application.grid_width,
        )

    @validator('theme', allow_reuse=True)
    def _valid_theme(cls, v):
        if not is_theme_available(v):
            raise ValueError(
                trans._(
                    "Theme '{theme_name}' not found; options are {themes}.",
                    deferred=True,
                    theme_name=v,
                    themes=', '.join(available_themes()),
                )
            )

        return v

    def json(self, **kwargs):
        """Serialize to json."""
        # Manually exclude the layer list and active layer which cannot be serialized at this point
        # and mouse and keybindings don't belong on model
        # https://github.com/samuelcolvin/pydantic/pull/2231
        # https://github.com/samuelcolvin/pydantic/issues/660#issuecomment-642211017
        exclude = kwargs.pop('exclude', set())
        exclude = exclude.union(EXCLUDE_JSON)
        return super().json(exclude=exclude, **kwargs)

    def dict(self, **kwargs):
        """Convert to a dictionary."""
        # Manually exclude the layer list and active layer which cannot be serialized at this point
        # and mouse and keybindings don't belong on model
        # https://github.com/samuelcolvin/pydantic/pull/2231
        # https://github.com/samuelcolvin/pydantic/issues/660#issuecomment-642211017
        exclude = kwargs.pop('exclude', set())
        exclude = exclude.union(EXCLUDE_DICT)
        return super().dict(exclude=exclude, **kwargs)

    def __hash__(self):
        return id(self)

    def __str__(self):
        """Simple string representation"""
        return f'napari.Viewer: {self.title}'

    @property
    def _sliced_extent_world_augmented(self) -> np.ndarray:
        """Extent of layers in world coordinates after slicing.

        D is either 2 or 3 depending on if the displayed data is 2D or 3D.

        Returns
        -------
        sliced_extent_world : array, shape (2, D)
        """
        # if not layers are present, assume image-like with dimensions of size 512
        if len(self.layers) == 0:
            return np.vstack(
                [np.full(self.dims.ndim, -0.5), np.full(self.dims.ndim, 511.5)]
            )
        return self.layers._extent_world_augmented[:, self.dims.displayed]

    def reset_view(
        self, *, margin: float = 0.05, reset_camera_angle: bool = True
    ) -> None:
        """Reset the camera view.

        Parameters
        ----------
        margin : float in [0, 1)
            Margin as fraction of the canvas, showing blank space around the
            data.
        """

        extent = self._sliced_extent_world_augmented
        scene_size = extent[1] - extent[0]
        corner = extent[0]
        grid_size = list(self.grid.actual_shape(len(self.layers)))
        if len(scene_size) > len(grid_size):
            grid_size = [1] * (len(scene_size) - len(grid_size)) + grid_size
        size = np.multiply(scene_size, grid_size)
        center_array = np.add(corner, np.divide(size, 2))[
            -self.dims.ndisplay :
        ]
        center = cast(
            Union[tuple[float, float, float], tuple[float, float]],
            tuple(
                [0.0] * (self.dims.ndisplay - len(center_array))
                + list(center_array)
            ),
        )
        assert len(center) in (2, 3)
        self.camera.center = center
        # zoom is defined as the number of canvas pixels per world pixel
        # The default value used below will zoom such that the whole field
        # of view will occupy 95% of the canvas on the most filled axis

        if 0 <= margin < 1:
            scale_factor = 1 - margin
        else:
            raise ValueError(
                trans._(
                    'margin must be between 0 and 1; got {margin} instead.',
                    deferred=True,
                    margin=margin,
                )
            )
        if np.max(size) == 0:
            self.camera.zoom = scale_factor * np.min(self._canvas_size)
        else:
            scale = np.array(size[-2:])
            scale[np.isclose(scale, 0)] = 1
            self.camera.zoom = scale_factor * np.min(
                np.array(self._canvas_size) / scale
            )
        if reset_camera_angle:
            self.camera.angles = (0, 0, 90)

        # Emit a reset view event, which is no longer used internally, but
        # which maybe useful for building on napari.
        self.events.reset_view(
            center=self.camera.center,
            zoom=self.camera.zoom,
            angles=self.camera.angles,
        )

    def _new_labels(self):
        """Create new labels layer filling full world coordinates space."""
        layers_extent = self.layers.extent
        extent = layers_extent.world
        scale = layers_extent.step
        scene_size = extent[1] - extent[0]
        corner = extent[0]
        shape = [
            np.round(s / sc).astype('int') + 1
            for s, sc in zip(scene_size, scale)
        ]
        dtype_str = get_settings().application.new_labels_dtype
        empty_labels = np.zeros(shape, dtype=dtype_str)
        self.add_labels(empty_labels, translate=np.array(corner), scale=scale)  # type: ignore[attr-defined]
        # We define `add_labels` dynamically, so mypy doesn't know about it.

    def _on_layer_reload(self, event: Event) -> None:
        self._layer_slicer.submit(
            layers=[event.layer], dims=self.dims, force=True
        )

    def _update_layers(self, *, layers=None):
        """Updates the contained layers.

        Parameters
        ----------
        layers : list of napari.layers.Layer, optional
            List of layers to update. If none provided updates all.
        """
        layers = layers or self.layers
        self._layer_slicer.submit(layers=layers, dims=self.dims)
        # If the currently selected layer is sliced asynchronously, then the value
        # shown with this position may be incorrect. See the discussion for more details:
        # https://github.com/napari/napari/pull/5377#discussion_r1036280855
        position = list(self.cursor.position)
        if len(position) < self.dims.ndim:
            # cursor dimensionality is outdated — reset to correct dimension
            position = [0.0] * self.dims.ndim
        for ind in self.dims.order[: -self.dims.ndisplay]:
            position[ind] = self.dims.point[ind]
        self.cursor.position = tuple(position)

    def _on_active_layer(self, event):
        """Update viewer state for a new active layer."""
        active_layer = event.value
        if active_layer is None:
            for layer in self.layers:
                layer.update_transform_box_visibility(False)
                layer.update_highlight_visibility(False)
            self.help = ''
            self.cursor.style = CursorStyle.STANDARD
            self.camera.mouse_pan = True
            self.camera.mouse_zoom = True
        else:
            active_layer.update_transform_box_visibility(True)
            active_layer.update_highlight_visibility(True)
            for layer in self.layers:
                if layer != active_layer:
                    layer.update_transform_box_visibility(False)
                    layer.update_highlight_visibility(False)
            self.help = active_layer.help
            self.cursor.style = active_layer.cursor
            self.cursor.size = active_layer.cursor_size
            self.camera.mouse_pan = active_layer.mouse_pan
            self.camera.mouse_zoom = active_layer.mouse_zoom
            self.update_status_from_cursor()

    @staticmethod
    def rounded_division(min_val, max_val, precision):
        warnings.warn(
            trans._(
                'Viewer.rounded_division is deprecated since v0.4.18 and will be removed in 0.6.0.'
            ),
            FutureWarning,
            stacklevel=2,
        )
        return int(((min_val + max_val) / 2) / precision) * precision

    def _on_layers_change(self):
        if len(self.layers) == 0:
            self.dims.ndim = 2
            self.dims.reset()
        else:
            ranges = self.layers._ranges
            # TODO: can be optimized with dims.update(), but events need fixing
            self.dims.ndim = len(ranges)
            self.dims.range = ranges

        new_dim = self.dims.ndim
        dim_diff = new_dim - len(self.cursor.position)
        if dim_diff < 0:
            self.cursor.position = self.cursor.position[:new_dim]
        elif dim_diff > 0:
            self.cursor.position = tuple(
                list(self.cursor.position) + [0] * dim_diff
            )
        self.events.layers_change()

    def _update_mouse_pan(self, event):
        """Set the viewer interactive mouse panning"""
        if event.source is self.layers.selection.active:
            self.camera.mouse_pan = event.mouse_pan

    def _update_mouse_zoom(self, event):
        """Set the viewer interactive mouse zoom"""
        if event.source is self.layers.selection.active:
            self.camera.mouse_zoom = event.mouse_zoom

    def _update_cursor(self, event):
        """Set the viewer cursor with the `event.cursor` string."""
        self.cursor.style = event.cursor

    def _update_cursor_size(self, event):
        """Set the viewer cursor_size with the `event.cursor_size` int."""
        self.cursor.size = event.cursor_size

    def _update_async(self, event: Event) -> None:
        """Set layer slicer to force synchronous if async is disabled."""
        self._layer_slicer._force_sync = not event.value

    def _calc_status_from_cursor(
        self,
    ) -> Optional[tuple[Union[str, Dict], str]]:
        if not self.mouse_over_canvas:
            return None
        active = self.layers.selection.active
        if active is not None and active._loaded:
            status = active.get_status(
                self.cursor.position,
                view_direction=self.cursor._view_direction,
                dims_displayed=list(self.dims.displayed),
                world=True,
            )

            if self.tooltip.visible:
                tooltip_text = active._get_tooltip_text(
                    np.asarray(self.cursor.position),
                    view_direction=np.asarray(self.cursor._view_direction),
                    dims_displayed=list(self.dims.displayed),
                    world=True,
                )
            else:
                tooltip_text = ''

            return status, tooltip_text

        return 'Ready', ''

    def update_status_from_cursor(self):
        """Update the status and tooltip from the cursor position."""
        status = self._calc_status_from_cursor()
        if status is not None:
            self.status, self.tooltip.text = status
        if (active := self.layers.selection.active) is not None:
            self.help = active.help

    def _on_grid_change(self):
        """Arrange the current layers is a 2D grid."""
        extent = self._sliced_extent_world_augmented
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            i_row, i_column = self.grid.position(n_layers - 1 - i, n_layers)
            self._subplot(layer, (i_row, i_column), extent)

    def _subplot(self, layer, position, extent):
        """Shift a layer to a specified position in a 2D grid.

        Parameters
        ----------
        layer : napari.layers.Layer
            Layer that is to be moved.
        position : 2-tuple of int
            New position of layer in grid.
        extent : array, shape (2, D)
            Extent of the world.
        """
        scene_shift = extent[1] - extent[0]
        translate_2d = np.multiply(scene_shift[-2:], position)
        translate = [0] * layer.ndim
        translate[-2:] = translate_2d
        layer._translate_grid = translate

    @property
    def experimental(self):
        """Experimental commands for IPython console.

        For example run "viewer.experimental.cmds.loader.help".
        """
        from napari.components.experimental.commands import (
            ExperimentalNamespace,
        )

        return ExperimentalNamespace(self.layers)

    def _on_add_layer(self, event):
        """Connect new layer events.

        Parameters
        ----------
        event : :class:`napari.layers.Layer`
            Layer to add.
        """
        layer = event.value

        # Connect individual layer events to viewer events
        # TODO: in a future PR, we should now be able to connect viewer *only*
        # to viewer.layers.events... and avoid direct viewer->layer connections
        layer.events.mouse_pan.connect(self._update_mouse_pan)
        layer.events.mouse_zoom.connect(self._update_mouse_zoom)
        layer.events.cursor.connect(self._update_cursor)
        layer.events.cursor_size.connect(self._update_cursor_size)
        layer.events.data.connect(self._on_layers_change)
        layer.events.scale.connect(self._on_layers_change)
        layer.events.translate.connect(self._on_layers_change)
        layer.events.rotate.connect(self._on_layers_change)
        layer.events.shear.connect(self._on_layers_change)
        layer.events.affine.connect(self._on_layers_change)
        layer.events.name.connect(self.layers._update_name)
        layer.events.reload.connect(self._on_layer_reload)
        if hasattr(layer.events, 'mode'):
            layer.events.mode.connect(self._on_layer_mode_change)
        self._layer_help_from_mode(layer)

        # Update dims and grid model
        self._on_layers_change()
        self._on_grid_change()
        # Slice current layer based on dims
        self._update_layers(layers=[layer])

        if len(self.layers) == 1:
            # set dims slider to the middle of all dimensions
            self.reset_view()
            self.dims._go_to_center_step()

    @staticmethod
    def _layer_help_from_mode(layer: Layer):
        """
        Update layer help text base on layer mode.
        """
        layer_to_func_and_mode: dict[type[Layer], list] = {
            Points: points_fun_to_mode,
            Labels: labels_fun_to_mode,
            Shapes: shapes_fun_to_mode,
            Vectors: vectors_fun_to_mode,
            Image: image_fun_to_mode,
            Surface: surface_fun_to_mode,
            Tracks: tracks_fun_to_mode,
        }

        help_li = []
        shortcuts = get_settings().shortcuts.shortcuts

        for fun, mode_ in layer_to_func_and_mode.get(layer.__class__, []):
            if mode_ == layer.mode:
                continue
            action_name = f'napari:{fun.__name__}'
            desc = action_manager._actions[action_name].description.lower()
            if not shortcuts.get(action_name, []):
                continue
            help_li.append(
                trans._(
                    'use <{shortcut}> for {desc}',
                    shortcut=shortcuts[action_name][0],
                    desc=desc,
                )
            )

        layer.help = ', '.join(help_li)

    def _on_layer_mode_change(self, event):
        self._layer_help_from_mode(event.source)
        if (active := self.layers.selection.active) is not None:
            self.help = active.help

    def _on_remove_layer(self, event):
        """Disconnect old layer events.

        Parameters
        ----------
        event : napari.utils.event.Event
            Event which will remove a layer.

        Returns
        -------
        layer : :class:`napari.layers.Layer` or list
            The layer that was added (same as input).
        """
        layer = event.value

        # Disconnect all connections from layer
        disconnect_events(layer.events, self)
        disconnect_events(layer.events, self.layers)

        # Clean up overlays
        for overlay in list(layer._overlays):
            del layer._overlays[overlay]

        self._on_layers_change()
        self._on_grid_change()

    def add_layer(self, layer: Layer) -> Layer:
        """Add a layer to the viewer.

        Parameters
        ----------
        layer : :class:`napari.layers.Layer`
            Layer to add.

        Returns
        -------
        layer : :class:`napari.layers.Layer` or list
            The layer that was added (same as input).
        """
        # Adding additional functionality inside `add_layer`
        # should be avoided to keep full functionality
        # from adding a layer through the `layers.append`
        # method
        self.layers.append(layer)
        return layer

    @rename_argument(
        from_name='interpolation',
        to_name='interpolation2d',
        version='0.6.0',
        since_version='0.4.17',
    )
    def add_image(
        self,
        data=None,
        *,
        channel_axis=None,
        affine=None,
        axis_labels=None,
        attenuation=0.05,
        blending=None,
        cache=True,
        colormap=None,
        contrast_limits=None,
        custom_interpolation_kernel_2d=None,
        depiction='volume',
        experimental_clipping_planes=None,
        gamma=1.0,
        interpolation2d='nearest',
        interpolation3d='linear',
        iso_threshold=None,
        metadata=None,
        multiscale=None,
        name=None,
        opacity=1.0,
        plane=None,
        projection_mode='none',
        rendering='mip',
        rgb=None,
        rotate=None,
        scale=None,
        shear=None,
        translate=None,
        units=None,
        visible=True,
    ) -> Union[Image, list[Image]]:
        """Add one or more Image layers to the layer list.

        Parameters
        ----------
        data : array or list of array
            Image data. Can be N >= 2 dimensional. If the last dimension has length
            3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
            list and arrays are decreasing in shape then the data is treated as
            a multiscale image. Please note multiscale rendering is only
            supported in 2D. In 3D, only the lowest resolution scale is
            displayed.
        channel_axis : int, optional
            Axis to expand image along. If provided, each channel in the data
            will be added as an individual image layer. In channel_axis mode,
            other parameters MAY be provided as lists. The Nth value of the list
            will be applied to the Nth channel in the data. If a single value
            is provided, it will be broadcast to all Layers.
            All parameters except data, rgb, and multiscale can be provided as
            list of values. If a list is provided, it must be the same length as
            the axis that is being expanded as channels.
        affine : n-D array or napari.utils.transforms.Affine
            (N+1, N+1) affine transformation matrix in homogeneous coordinates.
            The first (N, N) entries correspond to a linear transform and
            the final column is a length N translation vector and a 1 or a
            napari `Affine` transform object. Applied as an extra transform on
            top of the provided scale, rotate, and shear values.
        axis_labels : tuple of str
            Dimension names of the layer data.
            If not provided, axis_labels will be set to (..., 'axis -2', 'axis -1').
        attenuation : float or list of float
            Attenuation rate for attenuated maximum intensity projection.
        blending : str or list of str
            One of a list of preset blending modes that determines how RGB and
            alpha values of the layer visual get mixed. Allowed values are
            {'translucent', 'translucent_no_depth', 'additive', 'minimum', 'opaque'}.
        cache : bool or list of bool
            Whether slices of out-of-core datasets should be cached upon
            retrieval. Currently, this only applies to dask arrays.
        colormap : str, napari.utils.Colormap, tuple, dict, list or list of these types
            Colormaps to use for luminance images. If a string, it can be the name
            of a supported colormap from vispy or matplotlib or the name of
            a vispy color or a hexadecimal RGB color representation.
            If a tuple, the first value must be a string to assign as a name to a
            colormap and the second item must be a Colormap. If a dict, the key must
            be a string to assign as a name to a colormap and the value must be a
            Colormap.
        contrast_limits : list (2,)
            Intensity value limits to be used for determining the minimum and maximum colormap bounds for
            luminance images. If not passed, they will be calculated as the min and max intensity value of
            the image.
        custom_interpolation_kernel_2d : np.ndarray
            Convolution kernel used with the 'custom' interpolation mode in 2D rendering.
        depiction : str or list of str
            3D Depiction mode. Must be one of {'volume', 'plane'}.
            The default value is 'volume'.
        experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
            Each dict defines a clipping plane in 3D in data coordinates.
            Valid dictionary keys are {'position', 'normal', and 'enabled'}.
            Values on the negative side of the normal are discarded if the plane is enabled.
        gamma : float or list of float
            Gamma correction for determining colormap linearity; defaults to 1.
        interpolation2d : str or list of str
            Interpolation mode used by vispy for rendering 2d data.
            Must be one of our supported modes.
            (for list of supported modes see Interpolation enum)
            'custom' is a special mode for 2D interpolation in which a regular grid
            of samples is taken from the texture around a position using 'linear'
            interpolation before being multiplied with a custom interpolation kernel
            (provided with 'custom_interpolation_kernel_2d').
        interpolation3d : str or list of str
            Same as 'interpolation2d' but for 3D rendering.
        iso_threshold : float or list of float
            Threshold for isosurface.
        metadata : dict or list of dict
            Layer metadata.
        multiscale : bool
            Whether the data is a multiscale image or not. Multiscale data is
            represented by a list of array-like image data. If not specified by
            the user and if the data is a list of arrays that decrease in shape,
            then it will be taken to be multiscale. The first image in the list
            should be the largest. Please note multiscale rendering is only
            supported in 2D. In 3D, only the lowest resolution scale is
            displayed.
        name : str or list of str
            Name of the layer.
        opacity : float or list
            Opacity of the layer visual, between 0.0 and 1.0.
        plane : dict or SlicingPlane
            Properties defining plane rendering in 3D. Properties are defined in
            data coordinates. Valid dictionary keys are
            {'position', 'normal', 'thickness', and 'enabled'}.
        projection_mode : str
            How data outside the viewed dimensions, but inside the thick Dims slice will
            be projected onto the viewed dimensions. Must fit to cls._projectionclass
        rendering : str or list of str
            Rendering mode used by vispy. Must be one of our supported
            modes. If a list then must be same length as the axis that is being
            expanded as channels.
        rgb : bool, optional
            Whether the image is RGB or RGBA if rgb. If not
            specified by user, but the last dimension of the data has length 3 or 4,
            it will be set as `True`. If `False`, the image is interpreted as a
            luminance image.
        rotate : float, 3-tuple of float, n-D array or list.
            If a float, convert into a 2D rotation matrix using that value as an
            angle. If 3-tuple, convert into a 3D rotation matrix, using a yaw,
            pitch, roll convention. Otherwise, assume an nD rotation. Angles are
            assumed to be in degrees. They can be converted from radians with
            'np.degrees' if needed.
        scale : tuple of float or list of tuple of float
            Scale factors for the layer.
        shear : 1-D array or list.
            A vector of shear values for an upper triangular n-D shear matrix.
        translate : tuple of float or list of tuple of float
            Translation values for the layer.
        units : tuple of str or pint.Unit, optional
            Units of the layer data in world coordinates.
            If not provided, the default units are assumed to be pixels.
        visible : bool or list of bool
            Whether the layer visual is currently being displayed.

        Returns
        -------
        layer : :class:`napari.layers.Image` or list
            The newly-created image layer or list of image layers.
        """

        if colormap is not None:
            # standardize colormap argument(s) to Colormaps, and make sure they
            # are in AVAILABLE_COLORMAPS.  This will raise one of many various
            # errors if the colormap argument is invalid.  See
            # ensure_colormap for details
            if isinstance(colormap, list):
                colormap = [ensure_colormap(c) for c in colormap]
            else:
                colormap = ensure_colormap(colormap)

        # doing this here for IDE/console autocompletion in add_image function.
        kwargs = {
            'rgb': rgb,
            'axis_labels': axis_labels,
            'colormap': colormap,
            'contrast_limits': contrast_limits,
            'gamma': gamma,
            'interpolation2d': interpolation2d,
            'interpolation3d': interpolation3d,
            'rendering': rendering,
            'depiction': depiction,
            'iso_threshold': iso_threshold,
            'attenuation': attenuation,
            'name': name,
            'metadata': metadata,
            'scale': scale,
            'translate': translate,
            'rotate': rotate,
            'shear': shear,
            'affine': affine,
            'opacity': opacity,
            'blending': blending,
            'visible': visible,
            'multiscale': multiscale,
            'cache': cache,
            'plane': plane,
            'experimental_clipping_planes': experimental_clipping_planes,
            'custom_interpolation_kernel_2d': custom_interpolation_kernel_2d,
            'projection_mode': projection_mode,
            'units': units,
        }

        # these arguments are *already* iterables in the single-channel case.
        iterable_kwargs = {
            'scale',
            'translate',
            'rotate',
            'shear',
            'affine',
            'contrast_limits',
            'metadata',
            'experimental_clipping_planes',
            'custom_interpolation_kernel_2d',
            'axis_labels',
            'units',
        }

        if channel_axis is None:
            kwargs['colormap'] = kwargs['colormap'] or 'gray'
            kwargs['blending'] = kwargs['blending'] or 'translucent_no_depth'
            # Helpful message if someone tries to add multi-channel kwargs,
            # but forget the channel_axis arg
            for k, v in kwargs.items():
                if k not in iterable_kwargs and is_sequence(v):
                    raise TypeError(
                        trans._(
                            "Received sequence for argument '{argument}', did you mean to specify a 'channel_axis'? ",
                            deferred=True,
                            argument=k,
                        )
                    )
            layer = Image(data, **kwargs)
            self.layers.append(layer)

            return layer

        layerdata_list = split_channels(data, channel_axis, **kwargs)

        layer_list = [
            Image(image, **i_kwargs) for image, i_kwargs, _ in layerdata_list
        ]
        self.layers.extend(layer_list)

        return layer_list

    def open_sample(
        self,
        plugin: str,
        sample: str,
        reader_plugin: Optional[str] = None,
        **kwargs,
    ) -> list[Layer]:
        """Open `sample` from `plugin` and add it to the viewer.

        To see all available samples registered by plugins, use
        :func:`napari.plugins.available_samples`

        Parameters
        ----------
        plugin : str
            name of a plugin providing a sample
        sample : str
            name of the sample
        reader_plugin : str, optional
            reader plugin to use, passed to ``viewer.open``. Only used if the
            sample data is an URI (Uniform Resource Identifier). By default None.
        **kwargs
            additional kwargs will be passed to the sample data loader provided
            by `plugin`.  Use of ``**kwargs`` may raise an error if the kwargs do
            not match the sample data loader.

        Returns
        -------
        layers : list
            A list of any layers that were added to the viewer.

        Raises
        ------
        KeyError
            If `plugin` does not provide a sample named `sample`.
        """
        from napari.plugins import _npe2, plugin_manager

        plugin_spec_reader = None
        data: Union[None, SampleDataCreator, SampleData]
        # try with npe2
        data, available = _npe2.get_sample_data(plugin, sample)

        # then try with npe1
        if data is None:
            try:
                data = plugin_manager._sample_data[plugin][sample]['data']
            except KeyError:
                available += list(plugin_manager.available_samples())
        # npe2 uri sample data, extract the path so we can use viewer.open
        elif hasattr(data, '__self__') and hasattr(data.__self__, 'uri'):
            if (
                hasattr(data.__self__, 'reader_plugin')
                and data.__self__.reader_plugin != reader_plugin
            ):
                # if the user chose a reader_plugin, we use their choice
                # but we remember what the plugin declared so we can inform the user if it fails
                plugin_spec_reader = data.__self__.reader_plugin
                reader_plugin = reader_plugin or plugin_spec_reader
            data = data.__self__.uri

        if data is None:
            msg = trans._(
                'Plugin {plugin!r} does not provide sample data named {sample!r}. ',
                plugin=plugin,
                sample=sample,
                deferred=True,
            )
            if available:
                msg = trans._(
                    'Plugin {plugin!r} does not provide sample data named {sample!r}. Available samples include: {samples}.',
                    deferred=True,
                    plugin=plugin,
                    sample=sample,
                    samples=available,
                )
            else:
                msg = trans._(
                    'Plugin {plugin!r} does not provide sample data named {sample!r}. No plugin samples have been registered.',
                    deferred=True,
                    plugin=plugin,
                    sample=sample,
                )

            raise KeyError(msg)

        with layer_source(sample=(plugin, sample)):
            if callable(data):
                added = []
                for datum in data(**kwargs):
                    added.extend(self._add_layer_from_data(*datum))
                return added
            if isinstance(data, (str, Path)):
                try:
                    return self.open(data, plugin=reader_plugin)
                except Exception as e:
                    # user chose a different reader to the one specified by the plugin
                    # and it failed - let them know the plugin declared something else
                    if (
                        plugin_spec_reader is not None
                        and reader_plugin != plugin_spec_reader
                    ):
                        raise ValueError(
                            trans._(
                                'Chosen reader {chosen_reader} failed to open sample. Plugin {plugin} declares {original_reader} as the reader for this sample - try calling `open_sample` with no `reader_plugin` or passing {original_reader} explicitly.',
                                deferred=True,
                                plugin=plugin,
                                chosen_reader=reader_plugin,
                                original_reader=plugin_spec_reader,
                            )
                        ) from e
                    raise e  # noqa: TRY201

            raise TypeError(
                trans._(
                    'Got unexpected type for sample ({plugin!r}, {sample!r}): {data_type}',
                    deferred=True,
                    plugin=plugin,
                    sample=sample,
                    data_type=type(data),
                )
            )

    def open(
        self,
        path: PathOrPaths,
        *,
        stack: Union[bool, list[list[PathLike]]] = False,
        plugin: Optional[str] = 'napari',
        layer_type: Optional[LayerTypeName] = None,
        **kwargs,
    ) -> list[Layer]:
        """Open a path or list of paths with plugins, and add layers to viewer.

        A list of paths will be handed one-by-one to the napari_get_reader hook
        if stack is False, otherwise the full list is passed to each plugin
        hook.

        Parameters
        ----------
        path : str or list of str
            A filepath, directory, or URL (or a list of any) to open.
        stack : bool or list[list[str]], optional
            If a list of strings is passed as ``path`` and ``stack`` is ``True``, then the
            entire list will be passed to plugins.  It is then up to individual
            plugins to know how to handle a list of paths.  If ``stack`` is
            ``False``, then the ``path`` list is broken up and passed to plugin
            readers one by one.  by default False.
            If the stack option is a list of lists containing individual paths,
            the inner lists are passedto the reader and will be stacked.
        plugin : str, optional
            Name of a plugin to use, by default builtins.  If provided, will
            force ``path`` to be read with the specified ``plugin``.
            If None, ``plugin`` will be read from preferences or inferred if just
            one reader is compatible.
            If the requested plugin cannot read ``path``, an exception will be raised.
        layer_type : str, optional
            If provided, will force data read from ``path`` to be passed to the
            corresponding ``add_<layer_type>`` method (along with any
            additional) ``kwargs`` provided to this function.  This *may*
            result in exceptions if the data returned from the path is not
            compatible with the layer_type.
        **kwargs
            All other keyword arguments will be passed on to the respective
            ``add_layer`` method.

        Returns
        -------
        layers : list
            A list of any layers that were added to the viewer.
        """
        if plugin == 'builtins':
            warnings.warn(
                trans._(
                    'The "builtins" plugin name is deprecated and will not work in a future version. Please use "napari" instead.',
                    deferred=True,
                ),
            )
            plugin = 'napari'

        paths_: list[PathLike] = (
            [os.fspath(path)]
            if isinstance(path, (Path, str))
            else [os.fspath(p) for p in path]
        )

        paths: Sequence[PathOrPaths] = paths_
        # If stack is a bool and True, add an additional layer of nesting.
        if isinstance(stack, bool) and stack:
            paths = [paths_]
        # If stack is a list and True, extend the paths with the inner lists.
        elif isinstance(stack, list) and stack:
            paths = [paths_]
            paths.extend(stack)

        added: list[Layer] = []  # for layers that get added
        with progress(
            paths,
            desc=trans._('Opening Files'),
            total=(
                0 if len(paths) == 1 else None
            ),  # indeterminate bar for 1 file
        ) as pbr:
            for _path in pbr:
                # If _path is a list, set stack to True
                _stack = isinstance(_path, list)
                # If _path is not a list already, make it a list.
                _path = [_path] if not isinstance(_path, list) else _path
                if plugin:
                    added.extend(
                        self._add_layers_with_plugins(
                            _path,
                            kwargs=kwargs,
                            plugin=plugin,
                            layer_type=layer_type,
                            stack=_stack,
                        )
                    )
                # no plugin choice was made
                else:
                    layers = self._open_or_raise_error(
                        _path, kwargs, layer_type, _stack
                    )
                    added.extend(layers)

        return added

    def _open_or_raise_error(
        self,
        paths: list[Union[Path, str]],
        kwargs: Optional[Dict[str, Any]] = None,
        layer_type: Optional[LayerTypeName] = None,
        stack: bool = False,
    ):
        """Open paths if plugin choice is unambiguous, raising any errors.

        This function will open paths if there is no plugin choice to be made
        i.e. there is a preferred reader associated with this file extension,
        or there is only one plugin available. Any errors that occur during
        the opening process are raised. If multiple plugins
        are available to read these paths, an error is raised specifying
        this.

        Errors are also raised by this function when the given paths are not
        a list or tuple, or if no plugins are available to read the files.
        This assumes all files have the same extension, as other cases
        are not yet supported.

        This function is called from ViewerModel.open, which raises any
        errors returned. The QtViewer also calls this method but catches
        exceptions and opens a dialog for users to make a plugin choice.

        Parameters
        ----------
        paths : List[Path | str]
            list of file paths to open
        kwargs : Dict[str, Any], optional
            keyword arguments to pass to layer adding method, by default {}
        layer_type : Optional[str], optional
            layer type for paths, by default None
        stack : bool or list[list[str]], optional
            True if files should be opened as a stack, by default False.
            Can also be a list containing lists of files to stack.

        Returns
        -------
        added
            list of layers added
        plugin
            plugin used to try opening paths, if any

        Raises
        ------
        TypeError
            when paths is *not* a list or tuple
        NoAvailableReaderError
            when no plugins are available to read path
        ReaderPluginError
            when reading with only available or prefered plugin fails
        MultipleReaderError
            when multiple readers are available to read the path
        """
        paths = [os.fspath(path) for path in paths]  # PathObjects -> str
        _path = paths[0]
        # we want to display the paths nicely so make a help string here
        path_message = f'[{_path}], ...]' if len(paths) > 1 else _path
        readers = get_potential_readers(_path)
        if not readers:
            raise NoAvailableReaderError(
                trans._(
                    'No plugin found capable of reading {path_message}.',
                    path_message=path_message,
                    deferred=True,
                ),
                paths,
            )

        plugin = get_preferred_reader(_path)
        if plugin and plugin not in readers:
            warnings.warn(
                RuntimeWarning(
                    trans._(
                        "Can't find {plugin} plugin associated with {path_message} files. ",
                        plugin=plugin,
                        path_message=path_message,
                    )
                    + trans._(
                        "This may be because you've switched environments, or have uninstalled the plugin without updating the reader preference. "
                    )
                    + trans._(
                        'You can remove this preference in the preference dialog, or by editing `settings.plugins.extension2reader`.'
                    )
                )
            )
            plugin = None

        # preferred plugin exists, or we just have one plugin available
        if plugin or len(readers) == 1:
            plugin = plugin or next(iter(readers.keys()))
            try:
                added = self._add_layers_with_plugins(
                    paths,
                    kwargs=kwargs,
                    stack=stack,
                    plugin=plugin,
                    layer_type=layer_type,
                )
            # plugin failed
            except Exception as e:
                raise ReaderPluginError(
                    trans._(
                        'Tried opening with {plugin}, but failed.',
                        deferred=True,
                        plugin=plugin,
                    ),
                    plugin,
                    paths,
                ) from e
        # multiple plugins
        else:
            raise MultipleReaderError(
                trans._(
                    'Multiple plugins found capable of reading {path_message}. Select plugin from {plugins} and pass to reading function e.g. `viewer.open(..., plugin=...)`.',
                    path_message=path_message,
                    plugins=readers,
                    deferred=True,
                ),
                list(readers.keys()),
                paths,
            )

        return added

    def _add_layers_with_plugins(
        self,
        paths: list[PathLike],
        *,
        stack: bool,
        kwargs: Optional[Dict] = None,
        plugin: Optional[str] = None,
        layer_type: Optional[LayerTypeName] = None,
    ) -> list[Layer]:
        """Load a path or a list of paths into the viewer using plugins.

        This function is mostly called from self.open_path, where the ``stack``
        argument determines whether a list of strings is handed to plugins one
        at a time, or en-masse.

        Parameters
        ----------
        paths : list of str
            A filepath, directory, or URL (or a list of any) to open. If a
            list, the assumption is that the list is to be treated as a stack.
        kwargs : dict, optional
            keyword arguments that will be used to overwrite any of those that
            are returned in the meta dict from plugins.
        plugin : str, optional
            Name of a plugin to use.  If provided, will force ``path`` to be
            read with the specified ``plugin``.  If the requested plugin cannot
            read ``path``, an exception will be raised.
        layer_type : str, optional
            If provided, will force data read from ``path`` to be passed to the
            corresponding ``add_<layer_type>`` method (along with any
            additional) ``kwargs`` provided to this function.  This *may*
            result in exceptions if the data returned from the path is not
            compatible with the layer_type.
        stack : bool
            See `open` method
            Stack=False => path is unique string, and list of len(1)
            Stack=True => path is list of path

        Returns
        -------
        List[Layer]
            A list of any layers that were added to the viewer.
        """
        from napari.plugins.io import read_data_with_plugins

        assert stack is not None
        assert isinstance(paths, list)
        assert not isinstance(paths, str)
        for p in paths:
            assert isinstance(p, str)

        if stack:
            layer_data, hookimpl = read_data_with_plugins(
                paths, plugin=plugin, stack=stack
            )
        else:
            assert len(paths) == 1
            layer_data, hookimpl = read_data_with_plugins(
                paths, plugin=plugin, stack=stack
            )

        if layer_data is None:
            return []
        # glean layer names from filename. These will be used as *fallback*
        # names, if the plugin does not return a name kwarg in their meta dict.
        filenames: Iterator[PathLike]

        if len(paths) == len(layer_data):
            filenames = iter(paths)
        else:
            # if a list of paths has been returned as a list of layer data
            # without a 1:1 relationship between the two lists we iterate
            # over the first name
            filenames = itertools.repeat(paths[0])

        # add each layer to the viewer
        added: list[Layer] = []  # for layers that get added
        plugin = hookimpl.plugin_name if hookimpl else None
        for data, filename in zip(layer_data, filenames):
            basename, _ext = os.path.splitext(os.path.basename(filename))
            # actually add the layer
            if isinstance(data, Layer):
                data._set_source(Source(path=filename, reader_plugin=plugin))
                lyr = self.add_layer(data)
                current_added = [lyr]
            else:
                _data = _unify_data_and_user_kwargs(
                    data, kwargs, layer_type, fallback_name=basename
                )
                with layer_source(path=filename, reader_plugin=plugin):
                    current_added = self._add_layer_from_data(*_data)
            added.extend(current_added)
        return added

    def _add_layer_from_data(
        self,
        data,
        meta: Optional[Mapping[str, Any]] = None,
        layer_type: Optional[str] = None,
    ) -> list[Layer]:
        """Add arbitrary layer data to the viewer.

        Primarily intended for usage by reader plugin hooks.

        Parameters
        ----------
        data : Any
            Data in a format that is valid for the corresponding `add_*` method
            of the specified ``layer_type``.
        meta : dict, optional
            Dict of keyword arguments that will be passed to the corresponding
            `add_*` method.  MUST NOT contain any keyword arguments that are
            not valid for the corresponding method.
        layer_type : str
            Type of layer to add.  MUST have a corresponding add_* method on
            on the viewer instance.  If not provided, the layer is assumed to
            be "image", unless data.dtype is one of (np.int32, np.uint32,
            np.int64, np.uint64), in which case it is assumed to be "labels".

        Returns
        -------
        layers : list of layers
            A list of layers added to the viewer.

        Raises
        ------
        ValueError
            If ``layer_type`` is not one of the recognized layer types.
        TypeError
            If any keyword arguments in ``meta`` are unexpected for the
            corresponding `add_*` method for this layer_type.

        Examples
        --------
        A typical use case might be to upack a tuple of layer data with a
        specified layer_type.

        >>> viewer = napari.Viewer()
        >>> data = (
        ...     np.random.random((10, 2)) * 20,
        ...     {'face_color': 'blue'},
        ...     'points',
        ... )
        >>> viewer._add_layer_from_data(*data)

        """
        if layer_type is None or layer_type == '':
            # assumes that big integer type arrays are likely labels.
            layer_type = guess_labels(data)
        else:
            layer_type = layer_type.lower()

        if meta is None:
            meta = {}

        if layer_type not in layers.NAMES:
            raise ValueError(
                trans._(
                    "Unrecognized layer_type: '{layer_type}'. Must be one of: {layer_names}.",
                    deferred=True,
                    layer_type=layer_type,
                    layer_names=layers.NAMES,
                )
            )
        if layer_type == 'image' and 'channel_axis' in meta:
            warnings.warn(
                'Adding layers with channel_axis is deprecated in 0.5.0 and will be removed in 0.6.0.',
                category=FutureWarning,
                stacklevel=3,
            )
            layer = self.add_image(data, **(meta or {}))
            if isinstance(layer, list):
                return layer
            return [layer]

        try:
            layer = Layer.create(data, {**meta}, layer_type)
            self.add_layer(layer)
        except TypeError as exc:
            if 'unexpected keyword argument' not in str(exc):
                raise
            bad_key = str(exc).split('keyword argument ')[-1]
            raise TypeError(
                trans._(
                    '_add_layer_from_data received an unexpected keyword argument ({bad_key}) for layer type {layer_type}',
                    deferred=True,
                    bad_key=bad_key,
                    layer_type=layer_type,
                )
            ) from exc
        return [layer]


def _normalize_layer_data(data: LayerData) -> FullLayerData:
    """Accepts any layerdata tuple, and returns a fully qualified tuple.

    Parameters
    ----------
    data : LayerData
        1-, 2-, or 3-tuple with (data, meta, layer_type).

    Returns
    -------
    FullLayerData
        3-tuple with (data, meta, layer_type)

    Raises
    ------
    ValueError
        If data has len < 1 or len > 3, or if the second item in ``data`` is
        not a ``dict``, or the third item is not a valid layer_type ``str``
    """
    if not isinstance(data, tuple) and 0 < len(data) < 4:
        raise ValueError(
            trans._(
                'LayerData must be a 1-, 2-, or 3-tuple',
                deferred=True,
            )
        )

    _data = list(data)
    if len(_data) > 1:
        if not isinstance(_data[1], MutableMapping):
            raise ValueError(
                trans._(
                    'The second item in a LayerData tuple must be a dict or other MutableMapping.',
                    deferred=True,
                )
            )
    else:
        _data.append({})
    if len(_data) > 2:
        if _data[2] not in layers.NAMES:
            raise ValueError(
                trans._(
                    'The third item in a LayerData tuple must be one of: {layers!r}.',
                    deferred=True,
                    layers=layers.NAMES,
                )
            )
    else:
        _data.append(guess_labels(_data[0]))
    return tuple(_data)


def _unify_data_and_user_kwargs(
    data: LayerData,
    kwargs: Optional[dict] = None,
    layer_type: Optional[LayerTypeName] = None,
    fallback_name: Optional[str] = None,
) -> FullLayerData:
    """Merge data returned from plugins with options specified by user.

    If ``data == (data_, meta_, type_)``.  Then:

    - ``kwargs`` will be used to update ``meta_``
    - ``layer_type`` will replace ``type_`` and, if provided, ``meta_`` keys
        will be pruned to layer_type-appropriate kwargs
    - ``fallback_name`` is used if ``not meta_.get('name')``

    .. note:

        If a user specified both layer_type and additional keyword arguments
        to viewer.open(), it is their responsibility to make sure the kwargs
        match the layer_type.

    Parameters
    ----------
    data : LayerData
        1-, 2-, or 3-tuple with (data, meta, layer_type) returned from plugin.
    kwargs : dict, optional
        User-supplied keyword arguments, to override those in ``meta`` supplied
        by plugins.
    layer_type : str, optional
        A user-supplied layer_type string, to override the ``layer_type``
        declared by the plugin.
    fallback_name : str, optional
        A name for the layer, to override any name in ``meta`` supplied by the
        plugin.

    Returns
    -------
    FullLayerData
        Fully qualified LayerData tuple with user-provided overrides.
    """
    data_, meta_, type_ = _normalize_layer_data(data)

    if layer_type:
        # the user has explicitly requested this be a certain layer type
        # strip any kwargs from the plugin that are no longer relevant
        meta_ = prune_kwargs(meta_, layer_type)
        type_ = layer_type

    if not isinstance(meta_, dict):
        meta_ = dict(meta_)

    if kwargs:
        # if user provided kwargs, use to override any meta dict values that
        # were returned by the plugin. We only prune kwargs if the user did
        # *not* specify the layer_type. This means that if a user specified
        # both layer_type and additional keyword arguments to viewer.open(),
        # it is their responsibility to make sure the kwargs match the
        # layer_type.
        meta_.update(prune_kwargs(kwargs, type_) if not layer_type else kwargs)

    if not meta_.get('name') and fallback_name:
        meta_['name'] = fallback_name
    return data_, meta_, type_


def prune_kwargs(kwargs: Mapping[str, Any], layer_type: str) -> dict[str, Any]:
    """Return copy of ``kwargs`` with only keys valid for ``add_<layer_type>``

    Parameters
    ----------
    kwargs : dict
        A key: value mapping where some or all of the keys are parameter names
        for the corresponding ``Viewer.add_<layer_type>`` method.
    layer_type : str
        The type of layer that is going to be added with these ``kwargs``.

    Returns
    -------
    pruned_kwargs : dict
        A key: value mapping where all of the keys are valid parameter names
        for the corresponding ``Viewer.add_<layer_type>`` method.

    Raises
    ------
    ValueError
        If ``ViewerModel`` does not provide an ``add_<layer_type>`` method
        for the provided ``layer_type``.

    Examples
    --------
    >>> test_kwargs = {
    ...     'scale': (0.75, 1),
    ...     'blending': 'additive',
    ...     'size': 10,
    ... }
    >>> prune_kwargs(test_kwargs, 'image')
    {'scale': (0.75, 1), 'blending': 'additive'}

    >>> # only labels has the ``num_colors`` argument
    >>> prune_kwargs(test_kwargs, 'points')
    {'scale': (0.75, 1), 'blending': 'additive', 'size': 10}
    """
    add_method = getattr(ViewerModel, 'add_' + layer_type, None)
    if not add_method or layer_type == 'layer':
        raise ValueError(
            trans._(
                'Invalid layer_type: {layer_type}',
                deferred=True,
                layer_type=layer_type,
            )
        )

    # get valid params for the corresponding add_<layer_type> method
    valid = valid_add_kwargs()[layer_type]
    return {k: v for k, v in kwargs.items() if k in valid}


@lru_cache(maxsize=1)
def valid_add_kwargs() -> dict[str, set[str]]:
    """Return a dict where keys are layer types & values are valid kwargs."""
    valid = {}
    for meth in dir(ViewerModel):
        if not meth.startswith('add_') or meth[4:] == 'layer':
            continue
        params = inspect.signature(getattr(ViewerModel, meth)).parameters
        valid[meth[4:]] = set(params) - {'self', 'kwargs'}
    return valid


for _layer in (
    layers.Labels,
    layers.Points,
    layers.Shapes,
    layers.Surface,
    layers.Tracks,
    layers.Vectors,
):
    func = create_add_method(_layer)
    setattr(ViewerModel, func.__name__, func)
