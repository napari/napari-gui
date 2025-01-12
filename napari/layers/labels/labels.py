import typing
import warnings
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import (
    Any,
    ClassVar,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import ndimage as ndi
from skimage.draw import polygon2mask

from napari.layers._data_protocols import LayerDataProtocol
from napari.layers._multiscale_data import MultiScaleData
from napari.layers._scalar_field.scalar_field import ScalarFieldBase
from napari.layers.base import Layer, no_op
from napari.layers.base._base_mouse_bindings import (
    highlight_box_handles,
    transform_with_box,
)
from napari.layers.image._image_utils import guess_multiscale
from napari.layers.image._slice import _ImageSliceResponse
from napari.layers.labels._labels_constants import (
    IsoCategoricalGradientMode,
    LabelColorMode,
    LabelsRendering,
    Mode,
)
from napari.layers.labels._labels_mouse_bindings import (
    BrushSizeOnMouseMove,
    draw,
    pick,
)
from napari.layers.labels._labels_utils import (
    expand_slice,
    get_contours,
    indices_in_shape,
    interpolate_coordinates,
    sphere_indices,
)
from napari.layers.utils.layer_utils import _FeatureTable
from napari.utils._dtype import normalize_dtype, vispy_texture_dtype
from napari.utils._indexing import elements_in_slice, index_in_slice
from napari.utils.colormaps import (
    direct_colormap,
    label_colormap,
)
from napari.utils.colormaps.colormap import (
    CyclicLabelColormap,
    LabelColormapBase,
    _normalize_label_colormap,
)
from napari.utils.colormaps.colormap_utils import shuffle_and_extend_colormap
from napari.utils.events import EmitterGroup, Event
from napari.utils.events.custom_types import Array
from napari.utils.misc import StringEnum, _is_array_type
from napari.utils.naming import magic_name
from napari.utils.status_messages import generate_layer_coords_status
from napari.utils.translations import trans

__all__ = ('Labels',)


class Labels(ScalarFieldBase):
    """Labels (or segmentation) layer.

    An image-like layer where every pixel contains an integer ID
    corresponding to the region it belongs to.

    Parameters
    ----------
    data : array or list of array
        Labels data as an array or multiscale. Must be integer type or bools.
        Please note multiscale rendering is only supported in 2D. In 3D, only
        the lowest resolution scale is displayed.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    axis_labels : tuple of str, optional
        Dimension names of the layer data.
        If not provided, axis_labels will be set to (..., 'axis -2', 'axis -1').
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    colormap : CyclicLabelColormap or DirectLabelColormap or None
        Colormap to use for the labels. If None, a random colormap will be
        used.
    depiction : str
        3D Depiction mode. Must be one of {'volume', 'plane'}.
        The default value is 'volume'.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a label and each column
        is a feature. The first row corresponds to the background label.
    iso_gradient_mode : str
        Method for calulating the gradient (used to get the surface normal) in the
        'iso_categorical' rendering mode. Must be one of {'fast', 'smooth'}.
        'fast' uses a simple finite difference gradient in x, y, and z. 'smooth' uses an
        isotropic Sobel gradient, which is smoother but more computationally expensive.
        The default value is 'fast'.
    metadata : dict
        Layer metadata.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    name : str
        Name of the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    projection_mode : str
        How data outside the viewed dimensions but inside the thick Dims slice will
        be projected onto the viewed dimensions
    properties : dict {str: array (N,)} or DataFrame
        Properties for each label. Each property should be an array of length
        N, where N is the number of labels, and the first property corresponds
        to background.
    rendering : str
        3D Rendering mode used by vispy. Must be one {'translucent', 'iso_categorical'}.
        'translucent' renders without lighting. 'iso_categorical' uses isosurface
        rendering to calculate lighting effects on labeled surfaces.
        The default value is 'iso_categorical'.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : tuple of float
        Scale factors for the layer.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    translate : tuple of float
        Translation values for the layer.
    units : tuple of str or pint.Unit, optional
        Units of the layer data in world coordinates.
        If not provided, the default units are assumed to be pixels.
    visible : bool
        Whether the layer visual is currently being displayed.

    Attributes
    ----------
    data : array or list of array
        Integer label data as an array or multiscale. Can be N dimensional.
        Every pixel contains an integer ID corresponding to the region it
        belongs to. The label 0 is rendered as transparent. Please note
        multiscale rendering is only supported in 2D. In 3D, only
        the lowest resolution scale is displayed.
    axis_labels : tuple of str
        Dimension names of the layer data.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. The first image in the
        list should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    metadata : dict
        Labels metadata.
    num_colors : int
        Number of unique colors to use in colormap. DEPRECATED: set
        ``colormap`` directly, using `napari.utils.colormaps.label_colormap`.
    features : Dataframe-like
        Features table where each row corresponds to a label and each column
        is a feature. The first row corresponds to the background label.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each label. Each property should be an array of length
        N, where N is the number of labels, and the first property corresponds
        to background.
    color : dict of int to str or array
        Custom label to color mapping. Values must be valid color names or RGBA
        arrays. While there is no limit to the number of custom labels, the
        the layer will render incorrectly if they map to more than 1024 distinct
        colors. DEPRECATED: set ``colormap`` directly, using
        `napari.utils.colormaps.DirectLabelColormap`.
    seed : float
        Seed for colormap random generator. DEPRECATED: set ``colormap``
        directly, using `napari.utils.colormaps.label_colormap`.
    opacity : float
        Opacity of the labels, must be between 0 and 1.
    contiguous : bool
        If `True`, the fill bucket changes only connected pixels of same label.
    n_edit_dimensions : int
        The number of dimensions across which labels will be edited.
    contour : int
        If greater than 0, displays contours of labels instead of shaded regions
        with a thickness equal to its value. Must be >= 0.
    brush_size : float
        Size of the paint brush in data coordinates.
    iso_gradient_mode : str
        Method for calulating the gradient (used to get the surface normal) in the
        'iso_categorical' rendering mode. Must be one of {'fast', 'smooth'}.
        'fast' uses a simple finite difference gradient in x, y, and z. 'smooth' uses an
        isotropic Sobel gradient, which is smoother but more computationally expensive.
    selected_label : int
        Index of selected label. Can be greater than the current maximum label.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In PICK mode the cursor functions like a color picker, setting the
        clicked on label to be the current label. If the background is picked it
        will select the background label `0`.

        In PAINT mode the cursor functions like a paint brush changing any
        pixels it brushes over to the current label. If the background label
        `0` is selected than any pixels will be changed to background and this
        tool functions like an eraser. The size and shape of the cursor can be
        adjusted in the properties widget.

        In FILL mode the cursor functions like a fill bucket replacing pixels
        of the label clicked on with the current label. It can either replace
        all pixels of that label or just those that are contiguous with the
        clicked on pixel. If the background label `0` is selected than any
        pixels will be changed to background and this tool functions like an
        eraser.

        In ERASE mode the cursor functions similarly to PAINT mode, but to
        paint with background label, which effectively removes the label.
    plane : SlicingPlane
        Properties defining plane rendering in 3D.
    experimental_clipping_planes : ClippingPlaneList
        Clipping planes defined in data coordinates, used to clip the volume.
    units: tuple of pint.Unit
        Units of the layer data in world coordinates.

    Notes
    -----
    _selected_color : 4-tuple or None
        RGBA tuple of the color of the selected label, or None if the
        background label `0` is selected.
    """

    events: EmitterGroup
    _colormap: LabelColormapBase

    _modeclass = Mode

    _drag_modes: ClassVar[dict[Mode, Callable[['Labels', Event], None]]] = {  # type: ignore[assignment]
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: transform_with_box,
        Mode.PICK: pick,
        Mode.PAINT: draw,
        Mode.FILL: draw,
        Mode.ERASE: draw,
        Mode.POLYGON: no_op,  # the overlay handles mouse events in this mode
    }

    brush_size_on_mouse_move = BrushSizeOnMouseMove(min_brush_size=1)

    _move_modes: ClassVar[
        dict[StringEnum, Callable[['Labels', Event], None]]
    ] = {  # type: ignore[assignment]
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: highlight_box_handles,
        Mode.PICK: no_op,
        Mode.PAINT: brush_size_on_mouse_move,
        Mode.FILL: no_op,
        Mode.ERASE: brush_size_on_mouse_move,
        Mode.POLYGON: no_op,  # the overlay handles mouse events in this mode
    }

    _cursor_modes: ClassVar[dict[Mode, str]] = {  # type: ignore[assignment]
        Mode.PAN_ZOOM: 'standard',
        Mode.TRANSFORM: 'standard',
        Mode.PICK: 'cross',
        Mode.PAINT: 'circle',
        Mode.FILL: 'cross',
        Mode.ERASE: 'circle',
        Mode.POLYGON: 'cross',
    }

    _history_limit = 100

    def __init__(
        self,
        data,
        *,
        affine=None,
        axis_labels=None,
        blending='translucent',
        cache=True,
        colormap=None,
        depiction='volume',
        experimental_clipping_planes=None,
        features=None,
        iso_gradient_mode=IsoCategoricalGradientMode.FAST.value,
        metadata=None,
        multiscale=None,
        name=None,
        opacity=0.7,
        plane=None,
        projection_mode='none',
        properties=None,
        rendering='iso_categorical',
        rotate=None,
        scale=None,
        shear=None,
        translate=None,
        units=None,
        visible=True,
    ) -> None:
        if name is None and data is not None:
            name = magic_name(data)

        self._seed = 0.5
        # We use 50 colors (49 + transparency) by default for historical
        # consistency. This may change in future versions.
        self._random_colormap = label_colormap(
            49, self._seed, background_value=0
        )
        self._original_random_colormap = self._random_colormap
        self._direct_colormap = direct_colormap(
            {0: 'transparent', None: 'black'}
        )
        self._colormap = self._random_colormap
        self._color_mode = LabelColorMode.AUTO
        self._show_selected_label = False
        self._contour = 0

        data = self._ensure_int_labels(data)

        super().__init__(
            data,
            affine=affine,
            axis_labels=axis_labels,
            blending=blending,
            cache=cache,
            depiction=depiction,
            experimental_clipping_planes=experimental_clipping_planes,
            rendering=rendering,
            metadata=metadata,
            multiscale=multiscale,
            name=name,
            scale=scale,
            shear=shear,
            plane=plane,
            opacity=opacity,
            projection_mode=projection_mode,
            rotate=rotate,
            translate=translate,
            units=units,
            visible=visible,
        )

        self.events.add(
            brush_shape=Event,
            brush_size=Event,
            colormap=Event,
            contiguous=Event,
            contour=Event,
            features=Event,
            iso_gradient_mode=Event,
            labels_update=Event,
            n_edit_dimensions=Event,
            paint=Event,
            preserve_labels=Event,
            properties=Event,
            selected_label=Event,
            show_selected_label=Event,
        )

        from napari.components.overlays.labels_polygon import (
            LabelsPolygonOverlay,
        )

        self._overlays.update({'polygon': LabelsPolygonOverlay()})

        self._feature_table = _FeatureTable.from_layer(
            features=features, properties=properties
        )
        self._label_index = self._make_label_index()

        self._n_edit_dimensions = 2
        self._contiguous = True
        self._brush_size = 10

        self._iso_gradient_mode = IsoCategoricalGradientMode(iso_gradient_mode)

        self._selected_label = 1
        self.colormap.selection = self._selected_label
        self.colormap.use_selection = self._show_selected_label
        self._prev_selected_label = None
        self._selected_color = self.get_color(self._selected_label)
        self._updated_slice = None
        if colormap is not None:
            self._set_colormap(colormap)

        self._status = self.mode
        self._preserve_labels = False

    def _post_init(self):
        self._reset_history()
        # Trigger generation of view slice and thumbnail
        self.refresh()
        self._reset_editable()

    @property
    def rendering(self):
        """Return current rendering mode.

        Selects a preset rendering mode in vispy that determines how
        lablels are displayed.  Options include:

        * ``translucent``: voxel colors are blended along the view ray until
          the result is opaque.
        * ``iso_categorical``: isosurface for categorical data.
          Cast a ray until a non-background value is encountered. At that
          location, lighning calculations are performed to give the visual
          appearance of a surface.

        Returns
        -------
        str
            The current rendering mode
        """
        return str(self._rendering)

    @rendering.setter
    def rendering(self, rendering):
        self._rendering = LabelsRendering(rendering)
        self.events.rendering()

    @property
    def iso_gradient_mode(self) -> str:
        """Return current gradient mode for isosurface rendering.

        Selects the finite-difference gradient method for the isosurface shader. Options include:
            * ``fast``: use a simple finite difference gradient along each axis
            * ``smooth``: use an isotropic Sobel gradient, smoother but more
              computationally expensive

        Returns
        -------
        str
            The current gradient mode
        """
        return str(self._iso_gradient_mode)

    @iso_gradient_mode.setter
    def iso_gradient_mode(self, value: IsoCategoricalGradientMode | str):
        self._iso_gradient_mode = IsoCategoricalGradientMode(value)
        self.events.iso_gradient_mode()

    @property
    def contiguous(self):
        """bool: fill bucket changes only connected pixels of same label."""
        return self._contiguous

    @contiguous.setter
    def contiguous(self, contiguous):
        self._contiguous = contiguous
        self.events.contiguous()

    @property
    def n_edit_dimensions(self):
        return self._n_edit_dimensions

    @n_edit_dimensions.setter
    def n_edit_dimensions(self, n_edit_dimensions):
        self._n_edit_dimensions = n_edit_dimensions
        self.events.n_edit_dimensions()

    @property
    def contour(self) -> int:
        """int: displays contours of labels instead of shaded regions."""
        return self._contour

    @contour.setter
    def contour(self, contour: int) -> None:
        if contour < 0:
            raise ValueError('contour value must be >= 0')
        self._contour = int(contour)
        self.events.contour()
        self.refresh(extent=False)

    @property
    def brush_size(self):
        """float: Size of the paint in world coordinates."""
        return self._brush_size

    @brush_size.setter
    def brush_size(self, brush_size):
        self._brush_size = int(brush_size)
        self.cursor_size = self._calculate_cursor_size()
        self.events.brush_size()

    def _calculate_cursor_size(self):
        # Convert from brush size in data coordinates to
        # cursor size in world coordinates
        scale = self._data_to_world.scale
        min_scale = np.min(
            [abs(scale[d]) for d in self._slice_input.displayed]
        )
        return abs(self.brush_size * min_scale)

    def new_colormap(self, seed: int | None = None):
        if seed is None:
            seed = np.random.default_rng().integers(2**32 - 1)

        orig = self._original_random_colormap
        self.colormap = shuffle_and_extend_colormap(
            self._original_random_colormap, seed
        )
        self._original_random_colormap = orig

    @property
    def colormap(self) -> LabelColormapBase:
        return self._colormap

    @colormap.setter
    def colormap(self, colormap: LabelColormapBase):
        self._set_colormap(colormap)

    def _set_colormap(self, colormap):
        colormap = _normalize_label_colormap(colormap)
        if isinstance(colormap, CyclicLabelColormap):
            self._random_colormap = colormap
            self._original_random_colormap = colormap
            self._colormap = self._random_colormap
            color_mode = LabelColorMode.AUTO
        else:
            self._direct_colormap = colormap
            # `self._direct_colormap.color_dict` may contain just the default None and background label
            # colors, in which case we need to be in AUTO color mode. Otherwise,
            # `self._direct_colormap.color_dict` contains colors for all labels, and we should be in DIRECT
            # mode.

            # For more information
            # - https://github.com/napari/napari/issues/2479
            # - https://github.com/napari/napari/issues/2953
            if self._is_default_colors(self._direct_colormap.color_dict):
                color_mode = LabelColorMode.AUTO
                self._colormap = self._random_colormap
            else:
                color_mode = LabelColorMode.DIRECT
                self._colormap = self._direct_colormap
        self._cached_labels = None  # invalidate the cached color mapping
        self._selected_color = self.get_color(self.selected_label)
        self._color_mode = color_mode
        self.events.colormap()  # Will update the LabelVispyColormap shader
        self.events.selected_label()
        self.refresh(extent=False)

    @property
    def data(self) -> LayerDataProtocol | MultiScaleData:
        """array: Image data."""
        return self._data

    @data.setter
    def data(self, data: LayerDataProtocol | MultiScaleData):
        data = self._ensure_int_labels(data)
        self._data = data
        self._ndim = len(self._data.shape)
        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

    @property
    def features(self):
        """Dataframe-like features table.

        It is an implementation detail that this is a `pandas.DataFrame`. In the future,
        we will target the currently-in-development Data API dataframe protocol [1]_.
        This will enable us to use alternate libraries such as xarray or cuDF for
        additional features without breaking existing usage of this.

        If you need to specifically rely on the pandas API, please coerce this to a
        `pandas.DataFrame` using `features_to_pandas_dataframe`.

        References
        ----------
        .. [1] https://data-apis.org/dataframe-protocol/latest/API.html
        """
        return self._feature_table.values

    @features.setter
    def features(
        self,
        features: dict[str, np.ndarray] | pd.DataFrame,
    ) -> None:
        self._feature_table.set_values(features)
        self._label_index = self._make_label_index()
        self.events.properties()
        self.events.features()

    @property
    def properties(self) -> dict[str, np.ndarray]:
        """dict {str: array (N,)}, DataFrame: Properties for each label."""
        return self._feature_table.properties()

    @properties.setter
    def properties(self, properties: dict[str, Array]):
        self.features = properties

    def _make_label_index(self) -> dict[int, int]:
        features = self._feature_table.values
        label_index = {}
        if 'index' in features:
            label_index = {i: k for k, i in enumerate(features['index'])}
        elif features.shape[1] > 0:
            label_index = {i: i for i in range(features.shape[0])}
        return label_index

    def _is_default_colors(self, color):
        """Returns True if color contains only default colors, otherwise False.

        Default colors are black for `None` and transparent for
        `self.colormap.background_value`.

        Parameters
        ----------
        color : Dict
            Dictionary of label value to color array

        Returns
        -------
        bool
            True if color contains only default colors, otherwise False.
        """
        return (
            {None, self.colormap.background_value} == set(color.keys())
            and np.allclose(color[None], [0, 0, 0, 1])
            and np.allclose(
                color[self.colormap.background_value], [0, 0, 0, 0]
            )
        )

    def _ensure_int_labels(self, data):
        """Ensure data is integer by converting from bool if required, raising an error otherwise."""
        looks_multiscale, data = guess_multiscale(data)
        if not looks_multiscale:
            data = [data]
        int_data = []
        for data_level in data:
            # normalize_dtype turns e.g. tensorstore or torch dtypes into
            # numpy dtypes
            if np.issubdtype(normalize_dtype(data_level.dtype), np.floating):
                raise TypeError(
                    trans._(
                        'Only integer types are supported for Labels layers, but data contains {data_level_type}.',
                        data_level_type=data_level.dtype,
                    )
                )
            if data_level.dtype == bool:
                int_data.append(data_level.view(np.uint8))
            else:
                int_data.append(data_level)
        data = int_data
        if not looks_multiscale:
            data = data[0]
        return data

    def _get_state(self) -> dict[str, Any]:
        """Get dictionary of layer state.

        Returns
        -------
        state : dict of str to Any
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'multiscale': self.multiscale,
                'properties': self.properties,
                'rendering': self.rendering,
                'iso_gradient_mode': self.iso_gradient_mode,
                'depiction': self.depiction,
                'plane': self.plane.dict(),
                'experimental_clipping_planes': [
                    plane.dict() for plane in self.experimental_clipping_planes
                ],
                'data': self.data,
                'features': self.features,
                'colormap': self.colormap,
            }
        )
        return state

    @property
    def selected_label(self):
        """int: Index of selected label."""
        return self._selected_label

    @selected_label.setter
    def selected_label(self, selected_label):
        if selected_label == self.selected_label:
            return

        self._prev_selected_label = self.selected_label
        self.colormap.selection = selected_label
        self._selected_label = selected_label
        self._selected_color = self.get_color(selected_label)

        self.events.selected_label()

        if self.show_selected_label:
            self.refresh(extent=False)

    def swap_selected_and_background_labels(self):
        """Swap between the selected label and the background label."""
        if self.selected_label != self.colormap.background_value:
            self.selected_label = self.colormap.background_value
        else:
            self.selected_label = self._prev_selected_label

    @property
    def show_selected_label(self):
        """Whether to filter displayed labels to only the selected label or not"""
        return self._show_selected_label

    @show_selected_label.setter
    def show_selected_label(self, show_selected):
        self._show_selected_label = show_selected
        self.colormap.use_selection = show_selected
        self.colormap.selection = self.selected_label
        self.events.show_selected_label(show_selected_label=show_selected)
        self.refresh(extent=False)

    # Only overriding to change the docstring
    @property
    def mode(self):
        """MODE: Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In PICK mode the cursor functions like a color picker, setting the
        clicked on label to be the current label. If the background is picked it
        will select the background label `0`.

        In PAINT mode the cursor functions like a paint brush changing any
        pixels it brushes over to the current label. If the background label
        `0` is selected than any pixels will be changed to background and this
        tool functions like an eraser. The size and shape of the cursor can be
        adjusted in the properties widget.

        In FILL mode the cursor functions like a fill bucket replacing pixels
        of the label clicked on with the current label. It can either replace
        all pixels of that label or just those that are contiguous with the
        clicked on pixel. If the background label `0` is selected than any
        pixels will be changed to background and this tool functions like an
        eraser.

        In ERASE mode the cursor functions similarly to PAINT mode, but to
        paint with background label, which effectively removes the label.
        """
        return Layer.mode.fget(self)

    # Only overriding to change the docstring of the setter above
    @mode.setter
    def mode(self, mode):
        Layer.mode.fset(self, mode)

    def _mode_setter_helper(self, mode):
        mode = super()._mode_setter_helper(mode)
        if mode == self._mode:
            return mode

        self._overlays['polygon'].enabled = mode == Mode.POLYGON
        if mode in {Mode.PAINT, Mode.ERASE}:
            self.cursor_size = self._calculate_cursor_size()

        return mode

    @property
    def preserve_labels(self):
        """Defines if painting should preserve existing labels.

        Default to false to allow paint on existing labels. When
        set to true, existing labels will be preserved during painting.
        """
        return self._preserve_labels

    @preserve_labels.setter
    def preserve_labels(self, preserve_labels: bool):
        self._preserve_labels = preserve_labels
        self.events.preserve_labels(preserve_labels=preserve_labels)

    def _reset_editable(self) -> None:
        self.editable = not self.multiscale

    def _on_editable_changed(self) -> None:
        if not self.editable:
            self.mode = Mode.PAN_ZOOM
            self._reset_history()

    @staticmethod
    def _to_vispy_texture_dtype(data):
        """Convert data to a dtype that can be used as a VisPy texture.

        Labels layers allow all integer dtypes for data, but only a subset
        are supported by VisPy textures. For now, we convert all data to
        float32 as it can represent all input values (though not losslessly,
        see https://github.com/napari/napari/issues/6084).
        """
        return vispy_texture_dtype(data)

    def _update_slice_response(self, response: _ImageSliceResponse) -> None:
        """Override to convert raw slice data to displayed label colors."""
        response = response.to_displayed(self._raw_to_displayed)
        super()._update_slice_response(response)

    def _partial_labels_refresh(self):
        """Prepares and displays only an updated part of the labels."""

        if self._updated_slice is None or not self.loaded:
            return

        dims_displayed = self._slice_input.displayed
        raw_displayed = self._slice.image.raw

        # Keep only the dimensions that correspond to the current view
        updated_slice = tuple(
            self._updated_slice[index] for index in dims_displayed
        )

        offset = [axis_slice.start for axis_slice in updated_slice]

        if self.contour > 0:
            colors_sliced = self._raw_to_displayed(
                raw_displayed, data_slice=updated_slice
            )
        else:
            colors_sliced = self._slice.image.view[updated_slice]
        # The next line is needed to make the following tests pass in
        # napari/_vispy/_tests/:
        # - test_vispy_labels_layer.py::test_labels_painting
        # - test_vispy_labels_layer.py::test_labels_fill_slice
        # See https://github.com/napari/napari/pull/6112/files#r1291613760
        # and https://github.com/napari/napari/issues/6185
        self._slice.image.view[updated_slice] = colors_sliced

        self.events.labels_update(data=colors_sliced, offset=offset)
        self._updated_slice = None

    def _calculate_contour(
        self, labels: np.ndarray, data_slice: tuple[slice, ...]
    ) -> np.ndarray | None:
        """Calculate the contour of a given label array within the specified data slice.

        Parameters
        ----------
        labels : np.ndarray
            The label array.
        data_slice : Tuple[slice, ...]
            The slice of the label array on which to calculate the contour.

        Returns
        -------
        Optional[np.ndarray]
            The calculated contour as a boolean mask array.
            Returns None if the contour parameter is less than 1,
            or if the label array has more than 2 dimensions.
        """
        if self.contour < 1:
            return None
        if labels.ndim > 2:
            warnings.warn(
                trans._(
                    'Contours are not displayed during 3D rendering',
                    deferred=True,
                )
            )
            return None

        expanded_slice = expand_slice(data_slice, labels.shape, 1)
        sliced_labels = get_contours(
            labels[expanded_slice],
            self.contour,
            self.colormap.background_value,
        )

        # Remove the latest one-pixel border from the result
        delta_slice = tuple(
            slice(s1.start - s2.start, s1.stop - s2.start)
            for s1, s2 in zip(data_slice, expanded_slice, strict=False)
        )
        return sliced_labels[delta_slice]

    def _raw_to_displayed(
        self, raw, data_slice: tuple[slice, ...] | None = None
    ) -> np.ndarray:
        """Determine displayed image from a saved raw image and a saved seed.

        This function ensures that the 0 label gets mapped to the 0 displayed
        pixel.

        Parameters
        ----------
        raw : array or int
            Raw integer input image.

        data_slice : numpy array slice
            Slice that specifies the portion of the input image that
            should be computed and displayed.
            If None, the whole input image will be processed.

        Returns
        -------
        mapped_labels : array
            Encoded colors mapped between 0 and 1 to be displayed.
        """

        if data_slice is None:
            data_slice = tuple(slice(0, size) for size in raw.shape)

        labels = raw  # for readability

        sliced_labels = self._calculate_contour(labels, data_slice)

        # lookup function -> self._as_type

        if sliced_labels is None:
            sliced_labels = labels[data_slice]

        return self.colormap._data_to_texture(sliced_labels)

    def _update_thumbnail(self):
        """Update the thumbnail with current data and colormap.

        This is overridden from _ImageBase because we don't need to do things
        like adjusting gamma or changing the data based on the contrast
        limits.
        """
        if not self.loaded:
            # ASYNC_TODO: Do not compute the thumbnail until we are loaded.
            # Is there a nicer way to prevent this from getting called?
            return

        image = self._slice.thumbnail.raw
        if self._slice_input.ndisplay == 3 and self.ndim > 2:
            # we are only using the current slice so `image` will never be
            # bigger than 3. If we are in this clause, it is exactly 3, so we
            # use max projection. For labels, ideally we would use "first
            # nonzero projection", but we leave that for a future PR. (TODO)
            image = np.max(image, axis=0)
        imshape = np.array(image.shape[:2])
        thumbshape = np.array(self._thumbnail_shape[:2])

        raw_zoom_factor = np.min(thumbshape / imshape)
        new_shape = np.clip(
            raw_zoom_factor * imshape, a_min=1, a_max=thumbshape
        )
        zoom_factor = tuple(new_shape / imshape)

        downsampled = ndi.zoom(image, zoom_factor, prefilter=False, order=0)
        color_array = self.colormap.map(downsampled)
        color_array[..., 3] *= self.opacity

        self.thumbnail = color_array

    def get_color(self, label):
        """Return the color corresponding to a specific label."""
        if label == self.colormap.background_value:
            col = None
        elif label is None or (
            self.show_selected_label and label != self.selected_label
        ):
            col = self.colormap.map(self.colormap.background_value)
        else:
            col = self.colormap.map(label)
        return col

    def _reset_history(self, event=None):
        self._undo_history = deque(maxlen=self._history_limit)
        self._redo_history = deque(maxlen=self._history_limit)
        self._staged_history = []
        self._block_history = False

    @contextmanager
    def block_history(self):
        """Context manager to group history-editing operations together.

        While in the context, history atoms are grouped together into a
        "staged" history. When exiting the context, that staged history is
        committed to the undo history queue, and an event is emitted
        containing the change.
        """
        prev = self._block_history
        self._block_history = True
        try:
            yield
            self._commit_staged_history()
        finally:
            self._block_history = prev

    def _commit_staged_history(self):
        """Save staged history to undo history and clear it."""
        if self._staged_history:
            self._append_to_undo_history(self._staged_history)
            self._staged_history = []

    def _append_to_undo_history(self, item):
        """Append item to history and emit paint event.

        Parameters
        ----------
        item : List[Tuple[ndarray, ndarray, int]]
            list of history atoms to append to undo history.
        """
        self._undo_history.append(item)
        self.events.paint(value=item)

    def _save_history(self, value):
        """Save a history "atom" to the undo history.

        A history "atom" is a single change operation to the array. A history
        *item* is a collection of atoms that were applied together to make a
        single change. For example, when dragging and painting, at each mouse
        callback we create a history "atom", but we save all those atoms in
        a single history item, since we would want to undo one drag in one
        undo operation.

        Parameters
        ----------
        value : 3-tuple of arrays
            The value is a 3-tuple containing:

            - a numpy multi-index, pointing to the array elements that were
              changed
            - the values corresponding to those elements before the change
            - the value(s) after the change
        """
        self._redo_history.clear()
        if self._block_history:
            self._staged_history.append(value)
        else:
            self._append_to_undo_history([value])

    def _load_history(self, before, after, undoing=True):
        """Load a history item and apply it to the array.

        Parameters
        ----------
        before : list of history items
            The list of elements from which we want to load.
        after : list of history items
            The list of element to which to append the loaded element. In the
            case of an undo operation, this is the redo queue, and vice versa.
        undoing : bool
            Whether we are undoing (default) or redoing. In the case of
            redoing, we apply the "after change" element of a history element
            (the third element of the history "atom").

        See Also
        --------
        Labels._save_history
        """
        if len(before) == 0:
            return

        history_item = before.pop()
        after.append(list(reversed(history_item)))
        for prev_indices, prev_values, next_values in reversed(history_item):
            values = prev_values if undoing else next_values
            self.data[prev_indices] = values

        self.refresh()

    def undo(self):
        self._load_history(
            self._undo_history, self._redo_history, undoing=True
        )

    def redo(self):
        self._load_history(
            self._redo_history, self._undo_history, undoing=False
        )

    def fill(self, coord, new_label, refresh=True):
        """Replace an existing label with a new label, either just at the
        connected component if the `contiguous` flag is `True` or everywhere
        if it is `False`, working in the number of dimensions specified by
        the `n_edit_dimensions` flag.

        Parameters
        ----------
        coord : sequence of float
            Position of mouse cursor in image coordinates.
        new_label : int
            Value of the new label to be filled in.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        """
        int_coord = tuple(np.round(coord).astype(int))
        # If requested fill location is outside data shape then return
        if np.any(np.less(int_coord, 0)) or np.any(
            np.greater_equal(int_coord, self.data.shape)
        ):
            return

        # If requested new label doesn't change old label then return
        old_label = np.asarray(self.data[int_coord]).item()
        if old_label == new_label or (
            self.preserve_labels
            and old_label != self.colormap.background_value
        ):
            return

        dims_to_fill = sorted(
            self._slice_input.order[-self.n_edit_dimensions :]
        )
        data_slice_list = list(int_coord)
        for dim in dims_to_fill:
            data_slice_list[dim] = slice(None)
        data_slice = tuple(data_slice_list)
        labels = np.asarray(self.data[data_slice])
        slice_coord = tuple(int_coord[d] for d in dims_to_fill)

        matches = labels == old_label
        if self.contiguous:
            # if contiguous replace only selected connected component
            labeled_matches, num_features = ndi.label(matches)
            if num_features != 1:
                match_label = labeled_matches[slice_coord]
                matches = np.logical_and(
                    matches, labeled_matches == match_label
                )

        match_indices_local = np.nonzero(matches)
        if self.ndim not in {2, self.n_edit_dimensions}:
            n_idx = len(match_indices_local[0])
            match_indices = []
            j = 0
            for d in data_slice:
                if isinstance(d, slice):
                    match_indices.append(match_indices_local[j])
                    j += 1
                else:
                    match_indices.append(np.full(n_idx, d, dtype=np.intp))
        else:
            match_indices = match_indices_local

        match_indices = _coerce_indices_for_vectorization(
            self.data, match_indices
        )

        self.data_setitem(match_indices, new_label, refresh)

    def _draw(self, new_label, last_cursor_coord, coordinates):
        """Paint into coordinates, accounting for mode and cursor movement.

        The draw operation depends on the current mode of the layer.

        Parameters
        ----------
        new_label : int
            value of label to paint
        last_cursor_coord : sequence
            last painted cursor coordinates
        coordinates : sequence
            new cursor coordinates
        """
        if coordinates is None:
            return
        interp_coord = interpolate_coordinates(
            last_cursor_coord, coordinates, self.brush_size
        )
        for c in interp_coord:
            if (
                self._slice_input.ndisplay == 3
                and self.data[tuple(np.round(c).astype(int))] == 0
            ):
                continue
            if self._mode in [Mode.PAINT, Mode.ERASE]:
                self.paint(c, new_label, refresh=False)
            elif self._mode == Mode.FILL:
                self.fill(c, new_label, refresh=False)
        self._partial_labels_refresh()

    def paint(self, coord, new_label, refresh=True):
        """Paint over existing labels with a new label, using the selected
        brush shape and size, either only on the visible slice or in all
        n dimensions.

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        new_label : int
            Value of the new label to be filled in.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        """
        shape, dims_to_paint = self._get_shape_and_dims_to_paint()
        paint_scale = np.array(
            [self.scale[i] for i in dims_to_paint], dtype=float
        )

        slice_coord = [int(np.round(c)) for c in coord]
        if self.n_edit_dimensions < self.ndim:
            coord_paint = [coord[i] for i in dims_to_paint]
        else:
            coord_paint = coord

        # Ensure circle doesn't have spurious point
        # on edge by keeping radius as ##.5
        radius = np.floor(self.brush_size / 2) + 0.5
        mask_indices = sphere_indices(radius, tuple(paint_scale))

        mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(
            int
        )

        self._paint_indices(
            mask_indices, new_label, shape, dims_to_paint, slice_coord, refresh
        )

    def paint_polygon(self, points, new_label):
        """Paint a polygon over existing labels with a new label.

        Parameters
        ----------
        points : list of coordinates
            List of coordinates of the vertices of a polygon.
        new_label : int
            Value of the new label to be filled in.
        """
        shape, dims_to_paint = self._get_shape_and_dims_to_paint()

        if len(dims_to_paint) != 2:
            raise NotImplementedError(
                'Polygon painting is implemented only in 2D.'
            )

        points = np.array(points, dtype=int)
        slice_coord = points[0].tolist()
        points2d = points[:, dims_to_paint]

        polygon_mask = polygon2mask(shape, points2d)
        mask_indices = np.argwhere(polygon_mask)
        self._paint_indices(
            mask_indices,
            new_label,
            shape,
            dims_to_paint,
            slice_coord,
            refresh=True,
        )

    def _paint_indices(
        self,
        mask_indices,
        new_label,
        shape,
        dims_to_paint,
        slice_coord=None,
        refresh=True,
    ):
        """Paint over existing labels with a new label, using the selected
        mask indices, either only on the visible slice or in all n dimensions.

        Parameters
        ----------
        mask_indices : numpy array of integer coordinates
            Mask to paint represented by an array of its coordinates.
        new_label : int
            Value of the new label to be filled in.
        shape : list
            The label data shape upon which painting is performed.
        dims_to_paint : list
            List of dimensions of the label data that are used for painting.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        """
        dims_not_painted = sorted(
            self._slice_input.order[: -self.n_edit_dimensions]
        )
        # discard candidate coordinates that are out of bounds
        mask_indices = indices_in_shape(mask_indices, shape)

        # Transfer valid coordinates to slice_coord,
        # or expand coordinate if 3rd dim in 2D image
        slice_coord_temp = list(mask_indices.T)
        if self.n_edit_dimensions < self.ndim:
            for j, i in enumerate(dims_to_paint):
                slice_coord[i] = slice_coord_temp[j]
            for i in dims_not_painted:
                slice_coord[i] = slice_coord[i] * np.ones(
                    mask_indices.shape[0], dtype=int
                )
        else:
            slice_coord = slice_coord_temp

        slice_coord = _coerce_indices_for_vectorization(self.data, slice_coord)

        # slice coord is a tuple of coordinate arrays per dimension
        # subset it if we want to only paint into background/only erase
        # current label
        if self.preserve_labels:
            if new_label == self.colormap.background_value:
                keep_coords = self.data[slice_coord] == self.selected_label
            else:
                keep_coords = (
                    self.data[slice_coord] == self.colormap.background_value
                )
            slice_coord = tuple(sc[keep_coords] for sc in slice_coord)

        self.data_setitem(slice_coord, new_label, refresh)

    def _get_shape_and_dims_to_paint(self) -> tuple[list, list]:
        dims_to_paint = sorted(self._get_dims_to_paint())
        shape = list(self.data.shape)

        if self.n_edit_dimensions < self.ndim:
            shape = [shape[i] for i in dims_to_paint]

        return shape, dims_to_paint

    def _get_dims_to_paint(self) -> list:
        return list(self._slice_input.order[-self.n_edit_dimensions :])

    def _get_pt_not_disp(self) -> dict[int, int]:
        """
        Get indices of current visible slice.
        """
        slice_input = self._slice.slice_input
        point = np.round(
            self.world_to_data(slice_input.world_slice.point)
        ).astype(int)
        return {dim: point[dim] for dim in slice_input.not_displayed}

    def data_setitem(self, indices, value, refresh=True):
        """Set `indices` in `data` to `value`, while writing to edit history.

        Parameters
        ----------
        indices : tuple of arrays of int
            Indices in data to overwrite. Must be a tuple of arrays of length
            equal to the number of data dimensions. (Fancy indexing in [2]_).
        value : int or array of int
            New label value(s). If more than one value, must match or
            broadcast with the given indices.
        refresh : bool, default True
            whether to refresh the view, by default True

        References
        ----------
        .. [2] https://numpy.org/doc/stable/user/basics.indexing.html
        """
        changed_indices = self.data[indices] != value
        indices = tuple(x[changed_indices] for x in indices)

        if isinstance(value, Sequence):
            value = np.asarray(value, dtype=self._slice.image.raw.dtype)
        else:
            value = self._slice.image.raw.dtype.type(value)

        # Resize value array to remove unchanged elements
        if isinstance(value, np.ndarray):
            value = value[changed_indices]

        if not indices or indices[0].size == 0:
            return

        self._save_history(
            (
                indices,
                np.array(self.data[indices], copy=True),
                value,
            )
        )

        # update the labels image
        self.data[indices] = value

        pt_not_disp = self._get_pt_not_disp()
        displayed_indices = index_in_slice(
            indices, pt_not_disp, self._slice.slice_input.order
        )
        if isinstance(value, np.ndarray):
            visible_values = value[elements_in_slice(indices, pt_not_disp)]
        else:
            visible_values = value

        if not (  # if not a numpy array or numpy-backed xarray
            isinstance(self.data, np.ndarray)
            or isinstance(getattr(self.data, 'data', None), np.ndarray)
        ):
            # In the absence of slicing, the current slice becomes
            # invalidated by data_setitem; only in the special case of a NumPy
            # array, or a NumPy-array-backed Xarray, is the slice a view and
            # therefore updated automatically.
            # For other types, we update it manually here.
            self._slice.image.raw[displayed_indices] = visible_values

        # tensorstore and xarray do not return their indices in
        # np.ndarray format, so they need to be converted explicitly
        if not isinstance(self.data, np.ndarray):
            indices = [np.array(x).flatten() for x in indices]

        updated_slice = tuple(
            [
                slice(min(axis_indices), max(axis_indices) + 1)
                for axis_indices in indices
            ]
        )

        if self.contour > 0:
            # Expand the slice by 1 pixel as the changes can go beyond
            # the original slice because of the morphological dilation
            # (1 pixel because get_contours always applies 1 pixel dilation)
            updated_slice = expand_slice(updated_slice, self.data.shape, 1)
        else:
            # update data view
            self._slice.image.view[displayed_indices] = (
                self.colormap._data_to_texture(visible_values)
            )

        if self._updated_slice is None:
            self._updated_slice = updated_slice
        else:
            self._updated_slice = tuple(
                [
                    slice(min(s1.start, s2.start), max(s1.stop, s2.stop))
                    for s1, s2 in zip(
                        updated_slice, self._updated_slice, strict=False
                    )
                ]
            )

        if refresh is True:
            self._partial_labels_refresh()

    def _calculate_value_from_ray(self, values):
        non_bg = values != self.colormap.background_value
        if not np.any(non_bg):
            return None
        return values[np.argmax(np.ravel(non_bg))]

    def get_status(
        self,
        position: npt.ArrayLike | None = None,
        *,
        view_direction: npt.ArrayLike | None = None,
        dims_displayed: list[int] | None = None,
        world: bool = False,
    ) -> dict:
        """Status message information of the data at a coordinate position.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        source_info : dict
            Dict containing a information that can be used in a status update.
        """
        if position is not None:
            value = self.get_value(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        else:
            value = None

        source_info = self._get_source_info()

        pos = position
        if pos is not None:
            pos = np.asarray(pos)[-self.ndim :]
        source_info['coordinates'] = generate_layer_coords_status(pos, value)

        # if this labels layer has properties
        properties = self._get_properties(
            position,
            view_direction=np.asarray(view_direction),
            dims_displayed=dims_displayed,
            world=world,
        )
        if properties:
            source_info['coordinates'] += '; ' + ', '.join(properties)

        return source_info

    def _get_tooltip_text(
        self,
        position,
        *,
        view_direction: np.ndarray | None = None,
        dims_displayed: list[int] | None = None,
        world: bool = False,
    ):
        """
        tooltip message of the data at a coordinate position.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        msg : string
            String containing a message that can be used as a tooltip.
        """
        return '\n'.join(
            self._get_properties(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        )

    def _get_properties(
        self,
        position,
        *,
        view_direction: np.ndarray | None = None,
        dims_displayed: list[int] | None = None,
        world: bool = False,
    ) -> list:
        if len(self._label_index) == 0 or self.features.shape[1] == 0:
            return []

        value = self.get_value(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        # if the cursor is not outside the image or on the background
        if value is None:
            return []

        label_value: int = typing.cast(
            int, value[1] if self.multiscale else value
        )
        if label_value not in self._label_index:
            return [trans._('[No Properties]')]

        idx = self._label_index[label_value]
        return [
            f'{k}: {v[idx]}'
            for k, v in self.features.items()
            if k != 'index'
            and len(v) > idx
            and v[idx] is not None
            and not (isinstance(v[idx], float) and np.isnan(v[idx]))
        ]


def _coerce_indices_for_vectorization(array, indices: list) -> tuple:
    """Coerces indices so that they can be used for vectorized indexing in the given data array."""
    if _is_array_type(array, 'xarray.DataArray'):
        # Fix indexing for xarray if necessary
        # See http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
        # for difference from indexing numpy
        try:
            import xarray as xr
        except ModuleNotFoundError:
            pass
        else:
            return tuple(xr.DataArray(i) for i in indices)
    return tuple(indices)
