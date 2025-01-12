from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
from vispy.scene import Node
from vispy.visuals import ImageVisual

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.utils.gl import fix_data_dtype
from napari._vispy.visuals.volume import Volume as VolumeNode
from napari.layers._scalar_field.scalar_field import ScalarFieldBase
from napari.utils.translations import trans


class ScalarFieldLayerNode(ABC):
    """Abstract base class for scalar field layer nodes."""

    @abstractmethod
    def __init__(self, node=None, texture_format='auto') -> None:
        raise NotImplementedError

    @abstractmethod
    def get_node(self, ndisplay: int, dtype: np.dtype | None = None) -> Node:
        """Return the appropriate node for the given ndisplay and dtype."""
        raise NotImplementedError


class VispyScalarFieldBaseLayer(VispyBaseLayer[ScalarFieldBase]):
    def __init__(
        self,
        layer: ScalarFieldBase,
        node=None,
        texture_format='auto',
        layer_node_class=ScalarFieldLayerNode,
    ) -> None:
        # Use custom node from caller, or our standard image/volume nodes.
        self._layer_node = layer_node_class(
            node, texture_format=texture_format
        )

        # Default to 2D (image) node.
        super().__init__(layer, self._layer_node.get_node(2))

        self._array_like = True

        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.depiction.connect(self._on_depiction_change)
        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.plane.events.position.connect(
            self._on_plane_position_change
        )
        self.layer.plane.events.thickness.connect(
            self._on_plane_thickness_change
        )
        self.layer.plane.events.normal.connect(self._on_plane_normal_change)
        self.layer.events.custom_interpolation_kernel_2d.connect(
            self._on_custom_interpolation_kernel_2d_change
        )

        # display_change is special (like data_change) because it requires a
        # self.reset(). This means that we have to call it manually. Also,
        # it must be called before reset in order to set the appropriate node
        # first
        self._on_display_change()
        self.reset()
        self._on_data_change()

    def _on_display_change(self, data=None) -> None:
        parent = self.node.parent
        self.node.parent = None
        ndisplay = self.layer._slice_input.ndisplay
        self.node = self._layer_node.get_node(
            ndisplay, getattr(data, 'dtype', None)
        )

        if data is None:
            texture_format = self.node.texture_format
            data = np.zeros(
                (1,) * ndisplay,
                dtype=get_dtype_from_vispy_texture_format(texture_format),
            )

        self.node.visible = not self.layer._slice.empty and self.layer.visible

        self.node.set_data(data)

        self.node.parent = parent
        self.node.order = self.order
        for overlay_visual in self.overlays.values():
            overlay_visual.node.parent = self.node
        self.reset()

    def _on_data_change(self) -> None:
        data = fix_data_dtype(self.layer._data_view)
        ndisplay = self.layer._slice_input.ndisplay

        node = self._layer_node.get_node(
            ndisplay, getattr(data, 'dtype', None)
        )

        if ndisplay > data.ndim:
            data = data.reshape((1,) * (ndisplay - data.ndim) + data.shape)

        # Check if data exceeds MAX_TEXTURE_SIZE and downsample
        if self.MAX_TEXTURE_SIZE_2D is not None and ndisplay == 2:
            data = self.downsample_texture(data, self.MAX_TEXTURE_SIZE_2D)
        elif self.MAX_TEXTURE_SIZE_3D is not None and ndisplay == 3:
            data = self.downsample_texture(data, self.MAX_TEXTURE_SIZE_3D)

        # Check if ndisplay has changed current node type needs updating
        if (ndisplay == 3 and not isinstance(node, VolumeNode)) or (
            (ndisplay == 2 and not isinstance(node, ImageVisual))
            or node != self.node
        ):
            self._on_display_change(data)
        else:
            node.set_data(data)
            node.visible = not self.layer._slice.empty and self.layer.visible

        # Call to update order of translation values with new dims:
        self._on_matrix_change()
        node.update()

    def _on_custom_interpolation_kernel_2d_change(self) -> None:
        if self.layer._slice_input.ndisplay == 2:
            self.node.custom_kernel = self.layer.custom_interpolation_kernel_2d

    def _on_rendering_change(self) -> None:
        if isinstance(self.node, VolumeNode):
            self.node.method = self.layer.rendering

    def _on_depiction_change(self) -> None:
        if isinstance(self.node, VolumeNode):
            self.node.raycasting_mode = str(self.layer.depiction)

    def _on_blending_change(self, event=None) -> None:
        super()._on_blending_change()

    def _on_plane_thickness_change(self) -> None:
        if isinstance(self.node, VolumeNode):
            self.node.plane_thickness = self.layer.plane.thickness

    def _on_plane_position_change(self) -> None:
        if isinstance(self.node, VolumeNode):
            self.node.plane_position = self.layer.plane.position

    def _on_plane_normal_change(self) -> None:
        if isinstance(self.node, VolumeNode):
            self.node.plane_normal = self.layer.plane.normal

    def _on_colormap_change(self, event=None) -> None:
        raise NotImplementedError

    def reset(self, event=None) -> None:
        super().reset()
        self._on_rendering_change()
        self._on_depiction_change()
        self._on_plane_position_change()
        self._on_plane_normal_change()
        self._on_plane_thickness_change()
        self._on_custom_interpolation_kernel_2d_change()

    def downsample_texture(
        self, data: np.ndarray, MAX_TEXTURE_SIZE: int
    ) -> np.ndarray:
        """Downsample data based on maximum allowed texture size.

        Parameters
        ----------
        data : array
            Data to be downsampled if needed.
        MAX_TEXTURE_SIZE : int
            Maximum allowed texture size.

        Returns
        -------
        data : array
            Data that now fits inside texture.
        """
        if np.any(np.greater(data.shape, MAX_TEXTURE_SIZE)):
            if self.layer.multiscale:
                raise ValueError(
                    trans._(
                        'Shape of individual tiles in multiscale {shape} cannot '
                        'exceed GL_MAX_TEXTURE_SIZE {texture_size}. Rendering is '
                        'currently in {ndisplay}D mode.',
                        deferred=True,
                        shape=data.shape,
                        texture_size=MAX_TEXTURE_SIZE,
                        ndisplay=self.layer._slice_input.ndisplay,
                    )
                )
            warnings.warn(
                trans._(
                    'data shape {shape} exceeds GL_MAX_TEXTURE_SIZE {texture_size}'
                    ' in at least one axis and will be downsampled.'
                    ' Rendering is currently in {ndisplay}D mode.',
                    deferred=True,
                    shape=data.shape,
                    texture_size=MAX_TEXTURE_SIZE,
                    ndisplay=self.layer._slice_input.ndisplay,
                )
            )
            downsample = np.ceil(
                np.divide(data.shape, MAX_TEXTURE_SIZE)
            ).astype(int)
            scale = np.ones(self.layer.ndim)
            for i, d in enumerate(self.layer._slice_input.displayed):
                scale[d] = downsample[i]

            # tile2data is a ScaleTransform thus is has a .scale attribute, but
            # mypy cannot know this.
            self.layer._transforms['tile2data'].scale = scale

            self._on_matrix_change()
            slices = tuple(slice(None, None, ds) for ds in downsample)
            data = data[slices]
        return data


_VISPY_FORMAT_TO_DTYPE: dict[str | None, np.dtype] = {
    'r8': np.dtype(np.uint8),
    'r16': np.dtype(np.uint16),
    'r32f': np.dtype(np.float32),
}
_DTYPE_TO_VISPY_FORMAT = {v: k for k, v in _VISPY_FORMAT_TO_DTYPE.items()}

# this is moved after reverse mapping is defined
# to always have non None values in _DTYPE_TO_VISPY_FORMAT
_VISPY_FORMAT_TO_DTYPE[None] = np.dtype(np.float32)


def get_dtype_from_vispy_texture_format(format_str: str) -> np.dtype:
    """Get the numpy dtype from a vispy texture format string.

    Parameters
    ----------
    format_str : str
        The vispy texture format string.

    Returns
    -------
    dtype : numpy.dtype
        The numpy dtype corresponding to the vispy texture format string.
    """
    return _VISPY_FORMAT_TO_DTYPE.get(format_str, np.dtype(np.float32))
