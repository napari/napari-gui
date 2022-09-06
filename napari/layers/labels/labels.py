import warnings
from collections import deque
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from napari.utils.misc import _is_array_type

from ...utils import config
from ...utils._dtype import normalize_dtype
from ...utils.colormaps import (
    color_dict_to_colormap,
    label_colormap,
    low_discrepancy_image,
)
from ...utils.events import Event
from ...utils.events.custom_types import Array
from ...utils.geometry import clamp_point_to_bounding_box
from ...utils.naming import magic_name
from ...utils.status_messages import generate_layer_coords_status
from ...utils.translations import trans
from ..base import no_op
from ..image._image_utils import guess_multiscale
from ..image.image import _ImageBase
from ..utils.color_transformations import transform_color
from ..utils.layer_utils import _FeatureTable
from ._labels_constants import LabelColorMode, LabelsRendering, Mode
from ._labels_mouse_bindings import draw, pick, toggle, toggled_draw
from ._labels_utils import (
    indices_in_shape,
    interpolate_coordinates,
    sphere_indices,
)


class Labels(_ImageBase):
    """Labels (or segmentation) layer.

    An image-like layer where every pixel contains an integer ID
    corresponding to the region it belongs to.

    Parameters
    ----------
    data : array or list of array
        Labels data as an array or multiscale. Must be integer type or bools.
        Please note multiscale rendering is only supported in 2D. In 3D, only
        the lowest resolution scale is displayed.
    num_colors : int
        Number of unique colors to use in colormap.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a label and each column
        is a feature. The first row corresponds to the background label.
    properties : dict {str: array (N,)} or DataFrame
        Properties for each label. Each property should be an array of length
        N, where N is the number of labels, and the first property corresponds
        to background.
    color : dict of int to str or array
        Custom label to color mapping. Values must be valid color names or RGBA
        arrays.
    seed : float
        Seed for colormap random generator.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    rendering : str
        3D Rendering mode used by vispy. Must be one {'translucent', 'iso_categorical'}.
        'translucent' renders without lighting. 'iso_categorical' uses isosurface
        rendering to calculate lighting effects on labeled surfaces.
        The default value is 'iso_categorical'.
    depiction : str
        3D Depiction mode. Must be one of {'volume', 'plane'}.
        The default value is 'volume'.
    visible : bool
        Whether the layer visual is currently being displayed.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.

    Attributes
    ----------
    data : array or list of array
        Integer label data as an array or multiscale. Can be N dimensional.
        Every pixel contains an integer ID corresponding to the region it
        belongs to. The label 0 is rendered as transparent. Please note
        multiscale rendering is only supported in 2D. In 3D, only
        the lowest resolution scale is displayed.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. The first image in the
        list should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    metadata : dict
        Labels metadata.
    num_colors : int
        Number of unique colors to use in colormap.
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
        colors.
    seed : float
        Seed for colormap random generator.
    opacity : float
        Opacity of the labels, must be between 0 and 1.
    contiguous : bool
        If `True`, the fill bucket changes only connected pixels of same label.
    n_edit_dimensions : int
        The number of dimensions across which labels will be edited.
    contour : int
        If greater than 0, displays contours of labels instead of shaded regions
        with a thickness equal to its value.
    brush_size : float
        Size of the paint brush in data coordinates.
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

    Notes
    -----
    _selected_color : 4-tuple or None
        RGBA tuple of the color of the selected label, or None if the
        background label `0` is selected.
    """

    _history_limit = 100

    def __init__(
        self,
        data,
        *,
        num_colors=50,
        features=None,
        properties=None,
        color=None,
        seed=0.5,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=0.7,
        blending='translucent',
        rendering='iso_categorical',
        depiction='volume',
        visible=True,
        multiscale=None,
        cache=True,
        plane=None,
        experimental_clipping_planes=None,
    ):
        if name is None and data is not None:
            name = magic_name(data)

        self._seed = seed
        self._background_label = 0
        self._num_colors = num_colors
        self._random_colormap = label_colormap(self.num_colors)
        self._all_vals = np.array([], dtype=float)
        self._color_mode = LabelColorMode.AUTO
        self._show_selected_label = False
        self._contour = 0

        data = self._ensure_int_labels(data)
        self._color_lookup_func = None

        super().__init__(
            data,
            rgb=False,
            colormap=self._random_colormap,
            contrast_limits=[0.0, 1.0],
            interpolation2d='nearest',
            interpolation3d='nearest',
            rendering=rendering,
            depiction=depiction,
            iso_threshold=0,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
            multiscale=multiscale,
            cache=cache,
            plane=plane,
            experimental_clipping_planes=experimental_clipping_planes,
        )

        self.events.add(
            preserve_labels=Event,
            properties=Event,
            n_edit_dimensions=Event,
            contiguous=Event,
            brush_size=Event,
            selected_label=Event,
            color_mode=Event,
            brush_shape=Event,
            contour=Event,
            features=Event,
            paint=Event,
        )

        self._feature_table = _FeatureTable.from_layer(
            features=features, properties=properties
        )
        self._label_index = self._make_label_index()

        self._n_edit_dimensions = 2
        self._contiguous = True
        self._brush_size = 10

        self._selected_label = 1
        self._selected_color = self.get_color(self._selected_label)
        self.color = color

        self._mode = Mode.PAN_ZOOM
        self._status = self.mode
        self._preserve_labels = False

        # setup states for 'fill_contour' mode
        self.toggle_draw = False
        self._previous_data = None
        self._drawcoords = []
        self._reset_history()

        # Trigger generation of view slice and thumbnail
        self._update_dims()
        self._set_editable()

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
    def contour(self):
        """int: displays contours of labels instead of shaded regions."""
        return self._contour

    @contour.setter
    def contour(self, contour):
        self._contour = contour
        self.events.contour()
        self.refresh()

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
        min_scale = np.min([abs(scale[d]) for d in self._dims_displayed])
        return abs(self.brush_size * min_scale)

    @property
    def seed(self):
        """float: Seed for colormap random generator."""
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        # invalidate _all_vals to trigger re-generation
        # in _raw_to_displayed
        self._all_vals = np.array([])
        self._selected_color = self.get_color(self.selected_label)
        self.refresh()
        self.events.selected_label()

    @_ImageBase.colormap.setter
    def colormap(self, colormap):
        super()._set_colormap(colormap)
        self._selected_color = self.get_color(self.selected_label)

    @property
    def num_colors(self):
        """int: Number of unique colors to use in colormap."""
        return self._num_colors

    @num_colors.setter
    def num_colors(self, num_colors):
        self._num_colors = num_colors
        self.colormap = label_colormap(num_colors)
        self.refresh()
        self._selected_color = self.get_color(self.selected_label)
        self.events.selected_label()

    @property
    def data(self):
        """array: Image data."""
        return self._data

    @data.setter
    def data(self, data):
        data = self._ensure_int_labels(data)
        self._data = data
        self._update_dims()
        self.events.data(value=self.data)
        self._set_editable()

    @property
    def features(self):
        """Dataframe-like features table.

        It is an implementation detail that this is a `pandas.DataFrame`. In the future,
        we will target the currently-in-development Data API dataframe protocol [1].
        This will enable us to use alternate libraries such as xarray or cuDF for
        additional features without breaking existing usage of this.

        If you need to specifically rely on the pandas API, please coerce this to a
        `pandas.DataFrame` using `features_to_pandas_dataframe`.

        References
        ----------
        .. [1]: https://data-apis.org/dataframe-protocol/latest/API.html
        """
        return self._feature_table.values

    @features.setter
    def features(
        self,
        features: Union[Dict[str, np.ndarray], pd.DataFrame],
    ) -> None:
        self._feature_table.set_values(features)
        self._label_index = self._make_label_index()
        self.events.properties()
        self.events.features()

    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """dict {str: array (N,)}, DataFrame: Properties for each label."""
        return self._feature_table.properties()

    @properties.setter
    def properties(self, properties: Dict[str, Array]):
        self.features = properties

    def _make_label_index(self) -> Dict[int, int]:
        features = self._feature_table.values
        label_index = {}
        if 'index' in features:
            label_index = {i: k for k, i in enumerate(features['index'])}
        elif features.shape[1] > 0:
            label_index = {i: i for i in range(features.shape[0])}
        return label_index

    @property
    def color(self):
        """dict: custom color dict for label coloring"""
        return self._color

    @color.setter
    def color(self, color):
        if not color:
            color = {}

        if self._background_label not in color:
            color[self._background_label] = 'transparent'
        if None not in color:
            color[None] = 'black'

        colors = {
            label: transform_color(color_str)[0]
            for label, color_str in color.items()
        }
        self._color = colors

        # `colors` may contain just the default None and background label
        # colors, in which case we need to be in AUTO color mode. Otherwise,
        # `colors` contains colors for all labels, and we should be in DIRECT
        # mode.

        # For more information
        # - https://github.com/napari/napari/issues/2479
        # - https://github.com/napari/napari/issues/2953
        if self._is_default_colors(colors):
            color_mode = LabelColorMode.AUTO
        else:
            color_mode = LabelColorMode.DIRECT

        self.color_mode = color_mode

    def _is_default_colors(self, color):
        """Returns True if color contains only default colors, otherwise False.

        Default colors are black for `None` and transparent for
        `self._background_label`.

        Parameters
        ----------
        color : Dict
            Dictionary of label value to color array

        Returns
        -------
        bool
            True if color contains only default colors, otherwise False.
        """
        if len(color) != 2:
            return False

        if not hasattr(self, '_color'):
            return False

        default_keys = [None, self._background_label]
        if set(default_keys) != set(color.keys()):
            return False

        for key in default_keys:
            if not np.allclose(self._color[key], color[key]):
                return False

        return True

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
                        "Only integer types are supported for Labels layers, but data contains {data_level_type}.",
                        data_level_type=data_level.dtype,
                    )
                )
            if data_level.dtype == bool:
                int_data.append(data_level.astype(np.int8))
            else:
                int_data.append(data_level)
        data = int_data
        if not looks_multiscale:
            data = data[0]
        return data

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'multiscale': self.multiscale,
                'num_colors': self.num_colors,
                'properties': self.properties,
                'rendering': self.rendering,
                'depiction': self.depiction,
                'plane': self.plane.dict(),
                'experimental_clipping_planes': [
                    plane.dict() for plane in self.experimental_clipping_planes
                ],
                'seed': self.seed,
                'data': self.data,
                'color': self.color,
                'features': self.features,
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

        self._selected_label = selected_label
        self._selected_color = self.get_color(selected_label)
        self.events.selected_label()

        # note: self.color_mode returns a string and this comparison fails,
        # so use self._color_mode
        if self.show_selected_label:
            self.refresh()

    @property
    def color_mode(self):
        """Color mode to change how color is represented.

        AUTO (default) allows color to be set via a hash function with a seed.

        DIRECT allows color of each label to be set directly by a color dict.
        """
        return str(self._color_mode)

    @color_mode.setter
    def color_mode(self, color_mode: Union[str, LabelColorMode]):
        color_mode = LabelColorMode(color_mode)
        if color_mode == LabelColorMode.DIRECT:
            custom_colormap, label_color_index = color_dict_to_colormap(
                self.color
            )
            super()._set_colormap(custom_colormap)
            self._label_color_index = label_color_index
        elif color_mode == LabelColorMode.AUTO:
            self._label_color_index = {}
            super()._set_colormap(self._random_colormap)

        else:
            raise ValueError(trans._("Unsupported Color Mode"))

        self._color_mode = color_mode
        self._selected_color = self.get_color(self.selected_label)
        self.events.color_mode()
        self.events.colormap()
        self.events.selected_label()
        self.refresh()

    @property
    def show_selected_label(self):
        """Whether to filter displayed labels to only the selected label or not"""
        return self._show_selected_label

    @show_selected_label.setter
    def show_selected_label(self, filter):
        self._show_selected_label = filter
        self.refresh()

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
        return str(self._mode)

    _drag_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: no_op,
        Mode.PICK: pick,
        Mode.PAINT: draw,
        Mode.FILL: draw,
        Mode.ERASE: draw,
        Mode.FILL_CONTOUR: toggle,
    }

    _move_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: no_op,
        Mode.PICK: no_op,
        Mode.PAINT: no_op,
        Mode.FILL: no_op,
        Mode.ERASE: no_op,
        Mode.FILL_CONTOUR: toggled_draw,
    }

    _cursor_modes = {
        Mode.PAN_ZOOM: 'standard',
        Mode.TRANSFORM: 'standard',
        Mode.PICK: 'cross',
        Mode.PAINT: 'circle',
        Mode.FILL: 'cross',
        Mode.ERASE: 'circle',
        Mode.FILL_CONTOUR: 'circle',
    }

    @mode.setter
    def mode(self, mode: Union[str, Mode]):
        mode, changed = self._mode_setter_helper(mode, Mode)
        if not changed:
            return

        if mode in {Mode.PAINT, Mode.ERASE}:
            self.cursor_size = self._calculate_cursor_size()

        self.events.mode(mode=mode)
        self.refresh()

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

    @property
    def contrast_limits(self):
        return self._contrast_limits

    @contrast_limits.setter
    def contrast_limits(self, value):
        # Setting contrast_limits of labels layers leads to wrong visualization of the layer
        if tuple(value) != (0, 1):
            raise AttributeError(
                trans._(
                    "Setting contrast_limits on labels layers is not allowed.",
                    deferred=True,
                )
            )
        self._contrast_limits = (0, 1)

    def _set_editable(self, editable=None):
        """Set editable mode based on layer properties."""
        if editable is None:
            self.editable = not self.multiscale

        if not self.editable:
            self.mode = Mode.PAN_ZOOM
            self._reset_history()

    def _lookup_with_low_discrepancy_image(self, im, selected_label=None):
        """Returns display version of im using low_discrepancy_image.

        Passes the image through low_discrepancy_image, only coloring
        selected_label if it's not None.

        Parameters
        ----------
        im : array or int
            Raw integer input image.
        selected_label : int, optional
            Value of selected label to color, by default None
        """
        if selected_label:
            image = np.where(
                im == selected_label,
                low_discrepancy_image(selected_label, self._seed),
                0,
            )
        else:
            image = np.where(im != 0, low_discrepancy_image(im, self._seed), 0)
        return image

    def _lookup_with_index(self, im, selected_label=None):
        """Returns display version of im using color lookup array by index

        Parameters
        ----------
        im : array or int
            Raw integer input image.
        selected_label : int, optional
            Value of selected label to color, by default None
        """
        if selected_label:
            if selected_label > len(self._all_vals):
                self._color_lookup_func = self._get_color_lookup_func(
                    im,
                    min(np.min(im), selected_label),
                    max(np.max(im), selected_label),
                )
            if (
                self._color_lookup_func
                == self._lookup_with_low_discrepancy_image
            ):
                image = self._color_lookup_func(im, selected_label)
            else:
                colors = np.zeros_like(self._all_vals)
                colors[selected_label] = low_discrepancy_image(
                    selected_label, self._seed
                )
                image = colors[im]
        else:
            try:
                image = self._all_vals[im]
            except IndexError:
                self._color_lookup_func = self._get_color_lookup_func(
                    im, np.min(im), np.max(im)
                )
                if (
                    self._color_lookup_func
                    == self._lookup_with_low_discrepancy_image
                ):
                    # revert to "classic" mode converting all pixels since we
                    # encountered a large value in the raw labels image
                    image = self._color_lookup_func(im, selected_label)
                else:
                    image = self._all_vals[im]
        return image

    def _get_color_lookup_func(self, data, min_label_val, max_label_val):
        """Returns function used for mapping label values to colors

        If array of [0..max(data)] would be larger than data,
        returns lookup_with_low_discrepancy_image, otherwise returns
        lookup_with_index

        Parameters
        ----------
        data : array
            labels data
        min_label_val : int
            minimum label value in data
        max_label_val : int
            maximum label value in data

        Returns
        -------
        lookup_func : function
            function to use for mapping label values to colors
        """

        # low_discrepancy_image is slow for large images, but large labels can
        # blow up memory usage of an index array of colors. If the index array
        # would be larger than the image, we go back to computing the low
        # discrepancy image on the whole input image. (Up to a minimum value of
        # 1kB.)
        min_label_val0 = min(min_label_val, 0)
        # +1 to allow indexing with max_label_val
        data_range = max_label_val - min_label_val0 + 1
        nbytes_low_discrepancy = low_discrepancy_image(np.array([0])).nbytes
        max_nbytes = max(data.nbytes, 1024)
        if data_range * nbytes_low_discrepancy > max_nbytes:
            return self._lookup_with_low_discrepancy_image
        else:
            if self._all_vals.size < data_range:
                new_all_vals = low_discrepancy_image(
                    np.arange(min_label_val0, max_label_val + 1), self._seed
                )
                self._all_vals = np.roll(new_all_vals, min_label_val0)
                self._all_vals[0] = 0
            return self._lookup_with_index

    def _raw_to_displayed(self, raw):
        """Determine displayed image from a saved raw image and a saved seed.

        This function ensures that the 0 label gets mapped to the 0 displayed
        pixel.

        Parameters
        ----------
        raw : array or int
            Raw integer input image.

        Returns
        -------
        image : array
            Image mapped between 0 and 1 to be displayed.
        """

        raw_modified = raw
        if self.contour > 0:
            if raw.ndim == 2:
                raw_modified = np.zeros_like(raw)
                struct_elem = ndi.generate_binary_structure(raw.ndim, 1)
                thickness = self.contour
                thick_struct_elem = ndi.iterate_structure(
                    struct_elem, thickness
                ).astype(bool)
                boundaries = ndi.grey_dilation(
                    raw, footprint=struct_elem
                ) != ndi.grey_erosion(raw, footprint=thick_struct_elem)
                raw_modified[boundaries] = raw[boundaries]
            elif raw.ndim > 2:
                warnings.warn(
                    trans._(
                        "Contours are not displayed during 3D rendering",
                        deferred=True,
                    )
                )
        if self._color_lookup_func is None:
            self._color_lookup_func = self._get_color_lookup_func(
                raw_modified, np.min(raw_modified), np.max(raw_modified)
            )
        if (
            not self.show_selected_label
            and self._color_mode == LabelColorMode.DIRECT
        ):
            u, inv = np.unique(raw_modified, return_inverse=True)
            image = np.array(
                [
                    self._label_color_index[x]
                    if x in self._label_color_index
                    else self._label_color_index[None]
                    for x in u
                ]
            )[inv].reshape(raw_modified.shape)
        elif (
            not self.show_selected_label
            and self._color_mode == LabelColorMode.AUTO
        ):
            image = self._color_lookup_func(raw_modified)
        elif (
            self.show_selected_label
            and self._color_mode == LabelColorMode.AUTO
        ):
            image = self._color_lookup_func(raw_modified, self._selected_label)
        elif (
            self.show_selected_label
            and self._color_mode == LabelColorMode.DIRECT
        ):
            selected = self._selected_label
            if selected not in self._label_color_index:
                selected = None
            index = self._label_color_index
            image = np.where(
                raw_modified == selected,
                index[selected],
                np.where(
                    raw_modified != self._background_label,
                    index[None],
                    index[self._background_label],
                ),
            )
        else:
            raise ValueError("Unsupported Color Mode")
        return image

    def new_colormap(self):
        self.seed = np.random.rand()

    def get_color(self, label):
        """Return the color corresponding to a specific label."""
        if label == 0:
            col = None
        elif label is None:
            col = self.colormap.map([0, 0, 0, 0])[0]
        else:
            val = self._raw_to_displayed(np.array([label]))
            col = self.colormap.map(val)[0]
        return col

    def _get_value_ray(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Optional[int]:
        """Get the first non-background value encountered along a ray.

        Parameters
        ----------
        start_point : np.ndarray
            (n,) array containing the start point of the ray in data coordinates.
        end_point : np.ndarray
            (n,) array containing the end point of the ray in data coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the viewer.

        Returns
        -------
        value : Optional[int]
            The first non-zero value encountered along the ray. If none
            was encountered or the viewer is in 2D mode, None is returned.
        """
        if start_point is None or end_point is None:
            return None
        if len(dims_displayed) == 3:
            # only use get_value_ray on 3D for now
            # we use dims_displayed because the image slice
            # has its dimensions  in th same order as the vispy
            # Volume
            start_point = start_point[dims_displayed]
            end_point = end_point[dims_displayed]
            sample_ray = end_point - start_point
            length_sample_vector = np.linalg.norm(sample_ray)
            n_points = int(2 * length_sample_vector)
            sample_points = np.linspace(
                start_point, end_point, n_points, endpoint=True
            )
            im_slice = self._slice.image.raw
            clamped = clamp_point_to_bounding_box(
                sample_points, self._display_bounding_box(dims_displayed)
            ).astype(int)
            values = im_slice[tuple(clamped.T)]
            nonzero_indices = np.flatnonzero(values)
            if len(nonzero_indices > 0):
                # if a nonzer0 value was found, return the first one
                return values[nonzero_indices[0]]

        return None

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Optional[int]:
        """Get the first non-background value encountered along a ray.

        Parameters
        ----------
        start_point : np.ndarray
            (n,) array containing the start point of the ray in data coordinates.
        end_point : np.ndarray
            (n,) array containing the end point of the ray in data coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the viewer.

        Returns
        -------
        value : int
            The first non-zero value encountered along the ray. If a
            non-zero value is not encountered, returns 0 (the background value).
        """
        return (
            self._get_value_ray(
                start_point=start_point,
                end_point=end_point,
                dims_displayed=dims_displayed,
            )
            or 0
        )

    def _reset_history(self, event=None):
        self._undo_history = deque(maxlen=self._history_limit)
        self._redo_history = deque(maxlen=self._history_limit)
        self._staged_history = []
        self._block_history = False

    def _reset_toggle_draw(self):
        self._drawcoords = []

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
            self.preserve_labels and old_label != self._background_label
        ):
            return

        dims_to_fill = sorted(self._dims_order[-self.n_edit_dimensions :])
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
        ndisplay = len(self._dims_displayed)
        interp_coord = interpolate_coordinates(
            last_cursor_coord, coordinates, self.brush_size
        )
        for c in interp_coord:
            if (
                ndisplay == 3
                and self.data[tuple(np.round(c).astype(int))] == 0
            ):
                continue
            if self._mode in {Mode.PAINT, Mode.ERASE, Mode.FILL_CONTOUR}:
                self.paint(c, new_label, refresh=False)
            elif self._mode == Mode.FILL:
                self.fill(c, new_label, refresh=False)
        self.refresh()

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
        shape = self.data.shape
        dims_to_paint = sorted(self._dims_order[-self.n_edit_dimensions :])
        dims_not_painted = sorted(self._dims_order[: -self.n_edit_dimensions])
        paint_scale = np.array(
            [self.scale[i] for i in dims_to_paint], dtype=float
        )

        slice_coord = [int(np.round(c)) for c in coord]
        if self.n_edit_dimensions < self.ndim:
            coord_paint = [coord[i] for i in dims_to_paint]
            shape = [shape[i] for i in dims_to_paint]
        else:
            coord_paint = coord

        # Ensure circle doesn't have spurious point
        # on edge by keeping radius as ##.5
        radius = np.floor(self.brush_size / 2) + 0.5
        mask_indices = sphere_indices(radius, tuple(paint_scale))

        mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(
            int
        )

        # discard candidate coordinates that are out of bounds
        mask_indices = indices_in_shape(mask_indices, shape)

        # Transfer valid coordinates to slice_coord,
        # or expand coordinate if 3rd dim in 2D image
        slice_coord_temp = [m for m in mask_indices.T]
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
            if new_label == self._background_label:
                keep_coords = self.data[slice_coord] == self.selected_label
            else:
                keep_coords = self.data[slice_coord] == self._background_label
            slice_coord = tuple(sc[keep_coords] for sc in slice_coord)

        self.data_setitem(slice_coord, new_label, refresh)

    def data_setitem(self, indices, value, refresh=True):
        """Set `indices` in `data` to `value`, while writing to edit history.

        Parameters
        ----------
        indices : tuple of int, slice, or sequence of int
            Indices in data to overwrite. Can be any valid NumPy indexing
            expression [1]_.
        value : int or array of int
            New label value(s). If more than one value, must match or
            broadcast with the given indices.
        refresh : bool, default True
            whether to refresh the view, by default True

        References
        ----------
        ..[1] https://numpy.org/doc/stable/user/basics.indexing.html
        """
        self._save_history(
            (
                indices,
                np.array(self.data[indices], copy=True),
                value,
            )
        )

        # update the labels image
        self.data[indices] = value
        if refresh is True:
            self.refresh()

    def get_status(
        self,
        position: Optional[Tuple] = None,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
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
        source_info['coordinates'] = generate_layer_coords_status(
            position, value
        )

        # if this labels layer has properties
        properties = self._get_properties(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        if properties:
            source_info['coordinates'] += "; " + ", ".join(properties)

        return source_info

    def _get_tooltip_text(
        self,
        position,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
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
        return "\n".join(
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
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[List[int]] = None,
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

        label_value = value[1] if self.multiscale else value
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


if config.async_octree:
    from ..image.experimental.octree_image import _OctreeImageBase

    class Labels(Labels, _OctreeImageBase):
        pass


def _coerce_indices_for_vectorization(array, indices: list) -> tuple:
    """Coerces indices so that they can be used for vectorized indexing in the given data array."""
    if _is_array_type(array, 'xarray.DataArray'):
        # Fix indexing for xarray if necessary
        # See http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
        # for difference from indexing numpy
        try:
            import xarray as xr

            return tuple(xr.DataArray(i) for i in indices)
        except ModuleNotFoundError:
            pass
    return tuple(indices)
