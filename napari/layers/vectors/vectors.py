import warnings
from copy import copy
from typing import Any

import numpy as np
import pandas as pd

from napari.layers.base import Layer
from napari.layers.utils._color_manager_constants import ColorMode
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.utils.color_manager import ColorManager
from napari.layers.utils.color_transformations import ColorType
from napari.layers.utils.layer_utils import _FeatureTable
from napari.layers.vectors._slice import (
    _VectorSliceRequest,
    _VectorSliceResponse,
)
from napari.layers.vectors._vector_utils import fix_data_vectors
from napari.layers.vectors._vectors_constants import (
    VectorsProjectionMode,
    VectorStyle,
)
from napari.utils.colormaps import Colormap, ValidColormapArg
from napari.utils.events import Event
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans


class Vectors(Layer):
    """
    Vectors layer renders lines onto the canvas.

    Parameters
    ----------
    data : (N, 2, D) or (N1, N2, ..., ND, D) array
        An (N, 2, D) array is interpreted as "coordinate-like" data and a
        list of N vectors with start point and projections of the vector in
        D dimensions. An (N1, N2, ..., ND, D) array is interpreted as
        "image-like" data where there is a length D vector of the
        projections at each pixel.
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
    edge_color : str
        Color of all of the vectors.
    edge_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to edge_color if a
        categorical attribute is used color the vectors.
    edge_colormap : str, napari.utils.Colormap
        Colormap to set vector color if a continuous attribute is used to set edge_color.
    edge_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    edge_width : float
        Width for all vectors in pixels.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    feature_defaults : dict[str, Any] or DataFrame
        The default value of each feature in a table with one row.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a vector and each column
        is a feature.
    length : float
        Multiplicative factor on projections for length of all vectors.
    metadata : dict
        Layer metadata.
    name : str
        Name of the layer.
    ndim : int
        Number of dimensions for vectors. When data is not None, ndim must be D.
        An empty vectors layer can be instantiated with arbitrary ndim.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    out_of_slice_display : bool
        If True, renders vectors not just in central plane but also slightly out of slice
        according to specified point marker size.
    projection_mode : str
        How data outside the viewed dimensions but inside the thick Dims slice will
        be projected onto the viewed dimenions.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each vector. Each property should be an array of length N,
        where N is the number of vectors.
    property_choices : dict {str: array (N,)}
        possible values for each property.
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

    vector_style : str
        One of a list of preset display modes that determines how vectors are displayed.
        Allowed values are {'line', 'triangle', and 'arrow'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Attributes
    ----------
    data : (N, 2, D) array
        The start point and projections of N vectors in D dimensions.
    axis_labels : tuple of str
        Dimension names of the layer data.
    features : Dataframe-like
        Features table where each row corresponds to a vector and each column
        is a feature.
    feature_defaults : DataFrame-like
        Stores the default value of each feature in a table with one row.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each vector. Each property should be an array of length N,
        where N is the number of vectors.
    edge_width : float
        Width for all vectors in pixels.
    vector_style : VectorStyle
        Determines how vectors are displayed.

        * ``VectorStyle.LINE``:
            Vectors are displayed as lines.
        * ``VectorStyle.TRIANGLE``:
            Vectors are displayed as triangles.
        * ``VectorStyle.ARROW``:
            Vectors are displayed as arrows.
    length : float
        Multiplicative factor on projections for length of all vectors.
    edge_color : str
        Color of all of the vectors.
    edge_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to edge_color if a
        categorical attribute is used color the vectors.
    edge_colormap : str, napari.utils.Colormap
        Colormap to set vector color if a continuous attribute is used to set edge_color.
    edge_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    out_of_slice_display : bool
        If True, renders vectors not just in central plane but also slightly out of slice
        according to specified point marker size.
    units: tuple of pint.Unit
        Units of the layer data in world coordinates.

    Notes
    -----
    _view_data : (M, 2, 2) array
        The start point and projections of N vectors in 2D for vectors whose
        start point is in the currently viewed slice.
    _view_face_color : (M, 4) np.ndarray
        colors for the M in view vectors
    _view_indices : (1, M) array
        indices for the M in view vectors
    _view_alphas : (M,) or float
        relative opacity for the M in view vectors
    _property_choices : dict {str: array (N,)}
        Possible values for the properties in Vectors.properties.
    _max_vectors_thumbnail : int
        The maximum number of vectors that will ever be used to render the
        thumbnail. If more vectors are present then they are randomly
        subsampled.
    """

    _projectionclass = VectorsProjectionMode

    # The max number of vectors that will ever be used to render the thumbnail
    # If more vectors are present then they are randomly subsampled
    _max_vectors_thumbnail = 1024

    def __init__(
        self,
        data=None,
        *,
        affine=None,
        axis_labels=None,
        blending='translucent',
        cache=True,
        edge_color='red',
        edge_color_cycle=None,
        edge_colormap='viridis',
        edge_contrast_limits=None,
        edge_width=1,
        experimental_clipping_planes=None,
        feature_defaults=None,
        features=None,
        length=1,
        metadata=None,
        name=None,
        ndim=None,
        opacity=0.7,
        out_of_slice_display=False,
        projection_mode='none',
        properties=None,
        property_choices=None,
        rotate=None,
        scale=None,
        shear=None,
        translate=None,
        units=None,
        vector_style='triangle',
        visible=True,
    ) -> None:
        if ndim is None and scale is not None:
            ndim = len(scale)

        data, ndim = fix_data_vectors(data, ndim)

        super().__init__(
            data,
            ndim,
            affine=affine,
            axis_labels=axis_labels,
            blending=blending,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
            name=name,
            metadata=metadata,
            opacity=opacity,
            projection_mode=projection_mode,
            rotate=rotate,
            scale=scale,
            shear=shear,
            translate=translate,
            units=units,
            visible=visible,
        )

        # events for non-napari calculations
        self.events.add(
            length=Event,
            edge_width=Event,
            edge_color=Event,
            vector_style=Event,
            edge_color_mode=Event,
            properties=Event,
            out_of_slice_display=Event,
            features=Event,
            feature_defaults=Event,
        )

        # Save the vector style params
        self._vector_style = VectorStyle(vector_style)
        self._edge_width = edge_width
        self._out_of_slice_display = out_of_slice_display

        self._length = float(length)

        self._data = data

        self._feature_table = _FeatureTable.from_layer(
            features=features,
            feature_defaults=feature_defaults,
            properties=properties,
            property_choices=property_choices,
            num_data=len(self.data),
        )

        self._edge = ColorManager._from_layer_kwargs(
            n_colors=len(self.data),
            colors=edge_color,
            continuous_colormap=edge_colormap,
            contrast_limits=edge_contrast_limits,
            categorical_colormap=edge_color_cycle,
            properties=(
                self.properties
                if self._data.size > 0
                else self._feature_table.currents()
            ),
        )

        # Data containing vectors in the currently viewed slice
        self._view_data = np.empty((0, 2, 2))
        self._view_indices = np.array([], dtype=int)
        self._view_alphas: float | np.ndarray = 1.0

        # now that everything is set up, make the layer visible (if set to visible)
        self.refresh()
        self.visible = visible

    @property
    def data(self) -> np.ndarray:
        """(N, 2, D) array: start point and projections of vectors."""
        return self._data

    @data.setter
    def data(self, vectors: np.ndarray):
        previous_n_vectors = len(self.data)

        self._data, _ = fix_data_vectors(vectors, self.ndim)
        n_vectors = len(self.data)

        # Adjust the props/color arrays when the number of vectors has changed
        with self.events.blocker_all(), self._edge.events.blocker_all():
            self._feature_table.resize(n_vectors)
            if n_vectors < previous_n_vectors:
                # If there are now fewer points, remove the size and colors of the
                # extra ones
                if len(self._edge.colors) > n_vectors:
                    self._edge._remove(
                        np.arange(n_vectors, len(self._edge.colors))
                    )

            elif n_vectors > previous_n_vectors:
                # If there are now more points, add the size and colors of the
                # new ones
                adding = n_vectors - previous_n_vectors
                self._edge._update_current_properties(
                    self._feature_table.currents()
                )
                self._edge._add(n_colors=adding)

        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

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
        features: dict[str, np.ndarray] | pd.DataFrame,
    ) -> None:
        self._feature_table.set_values(features, num_data=len(self.data))
        if self._edge.color_properties is not None:
            if self._edge.color_properties.name not in self.features:
                self._edge.color_mode = ColorMode.DIRECT
                self._edge.color_properties = None
                warnings.warn(
                    trans._(
                        'property used for edge_color dropped',
                        deferred=True,
                    ),
                    RuntimeWarning,
                )
            else:
                edge_color_name = self._edge.color_properties.name
                property_values = self.features[edge_color_name].to_numpy()
                self._edge.color_properties = {
                    'name': edge_color_name,
                    'values': property_values,
                    'current_value': self.feature_defaults[edge_color_name][0],
                }
        self.events.properties()
        self.events.features()

    @property
    def properties(self) -> dict[str, np.ndarray]:
        """dict {str: array (N,)}, DataFrame: Annotations for each point"""
        return self._feature_table.properties()

    @properties.setter
    def properties(self, properties: dict[str, Array]):
        self.features = properties

    @property
    def feature_defaults(self):
        """Dataframe-like with one row of feature default values.

        See `features` for more details on the type of this property.
        """
        return self._feature_table.defaults

    @feature_defaults.setter
    def feature_defaults(
        self, defaults: dict[str, Any] | pd.DataFrame
    ) -> None:
        self._feature_table.set_defaults(defaults)
        self.events.feature_defaults()

    @property
    def property_choices(self) -> dict[str, np.ndarray]:
        return self._feature_table.choices()

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
                'length': self.length,
                'edge_width': self.edge_width,
                'vector_style': self.vector_style,
                'edge_color': (
                    self.edge_color
                    if self.data.size
                    else [self._edge.current_color]
                ),
                'edge_color_cycle': self.edge_color_cycle,
                'edge_colormap': self.edge_colormap.dict(),
                'edge_contrast_limits': self.edge_contrast_limits,
                'data': self.data,
                'properties': self.properties,
                'property_choices': self.property_choices,
                'ndim': self.ndim,
                'features': self.features,
                'feature_defaults': self.feature_defaults,
                'out_of_slice_display': self.out_of_slice_display,
            }
        )
        return state

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self.data.shape[2]

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        if len(self.data) == 0:
            extrema = np.full((2, self.ndim), np.nan)
        else:
            # Convert from projections to endpoints using the current length
            data = copy(self.data)
            data[:, 1, :] = data[:, 0, :] + self.length * data[:, 1, :]
            maxs = np.max(data, axis=(0, 1))
            mins = np.min(data, axis=(0, 1))
            extrema = np.vstack([mins, maxs])
        return extrema

    @property
    def out_of_slice_display(self) -> bool:
        """bool: renders vectors slightly out of slice."""
        return self._out_of_slice_display

    @out_of_slice_display.setter
    def out_of_slice_display(self, out_of_slice_display: bool) -> None:
        self._out_of_slice_display = out_of_slice_display
        self.events.out_of_slice_display()
        self.refresh(extent=False)

    @property
    def edge_width(self) -> float:
        """float: Width for all vectors in pixels."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: float):
        self._edge_width = edge_width

        self.events.edge_width()
        self.refresh(extent=False)

    @property
    def vector_style(self) -> str:
        """Vectors display mode: Determines how vectors are displayed.

        VectorStyle.LINE
            Displays vectors as rectangular lines.
        VectorStyle.TRIANGLE
            Displays vectors as triangles.
        VectorStyle.ARROW
            Displays vectors as arrows.
        """
        return str(self._vector_style)

    @vector_style.setter
    def vector_style(self, vector_style: str):
        old_vector_style = self._vector_style
        self._vector_style = VectorStyle(vector_style)
        if self._vector_style != old_vector_style:
            self.events.vector_style()
            self.refresh(extent=False, thumbnail=False)

    @property
    def length(self) -> float:
        """float: Multiplicative factor for length of all vectors."""
        return self._length

    @length.setter
    def length(self, length: float):
        self._length = float(length)

        self.events.length()
        self.refresh()

    @property
    def edge_color(self) -> np.ndarray:
        """(1 x 4) np.ndarray: Array of RGBA edge colors (applied to all vectors)"""
        return self._edge.colors

    @edge_color.setter
    def edge_color(self, edge_color: ColorType):
        self._edge._set_color(
            color=edge_color,
            n_colors=len(self.data),
            properties=self.properties,
            current_properties=self._feature_table.currents(),
        )
        self.events.edge_color()

    def refresh_colors(self, update_color_mapping: bool = False):
        """Calculate and update edge colors if using a cycle or color map

        Parameters
        ----------
        update_color_mapping : bool
            If set to True, the function will recalculate the color cycle map
            or colormap (whichever is being used). If set to False, the function
            will use the current color cycle map or color map. For example, if you
            are adding/modifying vectors and want them to be colored with the same
            mapping as the other vectors (i.e., the new vectors shouldn't affect
            the color cycle map or colormap), set update_color_mapping=False.
            Default value is False.
        """
        self._edge._refresh_colors(self.properties, update_color_mapping)

    @property
    def edge_color_mode(self) -> ColorMode:
        """str: Edge color setting mode

        DIRECT (default mode) allows each vector to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
        """
        return self._edge.color_mode

    @edge_color_mode.setter
    def edge_color_mode(self, edge_color_mode: str | ColorMode):
        edge_color_mode = ColorMode(edge_color_mode)

        if edge_color_mode == ColorMode.DIRECT:
            self._edge_color_mode = edge_color_mode
        elif edge_color_mode in (ColorMode.CYCLE, ColorMode.COLORMAP):
            if self._edge.color_properties is not None:
                color_property = self._edge.color_properties.name
            else:
                color_property = ''
            if color_property == '':
                if self.properties:
                    color_property = next(iter(self.properties))
                    self._edge.color_properties = {
                        'name': color_property,
                        'values': self.features[color_property].to_numpy(),
                        'current_value': self.feature_defaults[color_property][
                            0
                        ],
                    }
                    warnings.warn(
                        trans._(
                            'edge_color property was not set, setting to: {color_property}',
                            deferred=True,
                            color_property=color_property,
                        ),
                        RuntimeWarning,
                    )
                else:
                    raise ValueError(
                        trans._(
                            'There must be a valid Points.properties to use {edge_color_mode}',
                            deferred=True,
                            edge_color_mode=edge_color_mode,
                        )
                    )

            # ColorMode.COLORMAP can only be applied to numeric properties
            if (edge_color_mode == ColorMode.COLORMAP) and not issubclass(
                self.properties[color_property].dtype.type,
                np.number,
            ):
                raise TypeError(
                    trans._(
                        'selected property must be numeric to use ColorMode.COLORMAP',
                        deferred=True,
                    )
                )

            self._edge.color_mode = edge_color_mode
        self.events.edge_color_mode()

    @property
    def edge_color_cycle(self) -> np.ndarray:
        """list, np.ndarray :  Color cycle for edge_color.
        Can be a list of colors defined by name, RGB or RGBA
        """
        return self._edge.categorical_colormap.fallback_color.values

    @edge_color_cycle.setter
    def edge_color_cycle(self, edge_color_cycle: list | np.ndarray):
        self._edge.categorical_colormap = edge_color_cycle

    @property
    def edge_colormap(self) -> Colormap:
        """Return the colormap to be applied to a property to get the edge color.

        Returns
        -------
        colormap : napari.utils.Colormap
            The Colormap object.
        """
        return self._edge.continuous_colormap

    @edge_colormap.setter
    def edge_colormap(self, colormap: ValidColormapArg):
        self._edge.continuous_colormap = colormap

    @property
    def edge_contrast_limits(self) -> tuple[float, float]:
        """None, (float, float): contrast limits for mapping
        the edge_color colormap property to 0 and 1
        """
        return self._edge.contrast_limits

    @edge_contrast_limits.setter
    def edge_contrast_limits(
        self, contrast_limits: None | tuple[float, float]
    ):
        self._edge.contrast_limits = contrast_limits

    @property
    def _view_face_color(self) -> np.ndarray:
        """(Mx4) np.ndarray : colors for the M in view triangles"""

        # Create as many colors as there are visible vectors.
        # Using fancy array indexing implicitly creates a new
        # array rather than creating a view of the original one
        # in ColorManager
        face_color = self.edge_color[self._view_indices]
        face_color[:, -1] *= self._view_alphas

        # Generally, several triangles are drawn for each vector,
        # so we need to duplicate the colors accordingly
        if self.vector_style == 'line':
            # Line vectors are drawn with 2 triangles
            face_color = np.repeat(face_color, 2, axis=0)

        elif self.vector_style == 'triangle':
            # Triangle vectors are drawn with 1 triangle
            pass  # No need to duplicate colors

        elif self.vector_style == 'arrow':
            # Arrow vectors are drawn with 3 triangles
            face_color = np.repeat(face_color, 3, axis=0)

        if self._slice_input.ndisplay == 3 and self.ndim > 2:
            face_color = np.vstack([face_color, face_color])

        return face_color

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        # The new slicing code makes a request from the existing state and
        # executes the request on the calling thread directly.
        # For async slicing, the calling thread will not be the main thread.
        request = self._make_slice_request_internal(
            self._slice_input, self._data_slice
        )
        response = request()
        self._update_slice_response(response)

    def _make_slice_request(self, dims) -> _VectorSliceRequest:
        """Make a Vectors slice request based on the given dims and these data."""
        slice_input = self._make_slice_input(dims)
        # TODO: [see Image]
        #   For the existing sync slicing, slice_indices is passed through
        # to avoid some performance issues related to the evaluation of the
        # data-to-world transform and its inverse. Async slicing currently
        # absorbs these performance issues here, but we can likely improve
        # things either by caching the world-to-data transform on the layer
        # or by lazily evaluating it in the slice task itself.
        slice_indices = slice_input.data_slice(self._data_to_world.inverse)
        return self._make_slice_request_internal(slice_input, slice_indices)

    def _make_slice_request_internal(
        self, slice_input: _SliceInput, data_slice: _ThickNDSlice
    ):
        return _VectorSliceRequest(
            slice_input=slice_input,
            data=self.data,
            data_slice=data_slice,
            projection_mode=self.projection_mode,
            out_of_slice_display=self.out_of_slice_display,
            length=self.length,
        )

    def _update_slice_response(self, response: _VectorSliceResponse):
        """Handle a slicing response."""
        self._slice_input = response.slice_input
        indices = response.indices
        alphas = response.alphas

        disp = self._slice_input.displayed

        self._view_indices = indices
        self._view_alphas = alphas
        self._view_data = self.data[np.ix_(list(indices), [0, 1], disp)]

    def _update_thumbnail(self):
        """Update thumbnail with current vectors and colors."""
        # Set the default thumbnail to black, opacity 1
        colormapped = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        colormapped[..., 3] = 1
        if len(self.data) == 0:
            self.thumbnail = colormapped
        else:
            # calculate min vals for the vertices and pad with 0.5
            # the offset is needed to ensure that the top left corner of the
            # vectors corresponds to the top left corner of the thumbnail
            de = self._extent_data
            offset = (
                np.array([de[0, d] for d in self._slice_input.displayed]) + 0.5
            )[-2:]
            # calculate range of values for the vertices and pad with 1
            # padding ensures the entire vector can be represented in the thumbnail
            # without getting clipped
            shape = np.ceil(
                [de[1, d] - de[0, d] + 1 for d in self._slice_input.displayed]
            ).astype(int)[-2:]
            zoom_factor = np.divide(self._thumbnail_shape[:2], shape).min()

            if self._view_data.shape[0] > self._max_vectors_thumbnail:
                thumbnail_indices = np.random.randint(
                    0, self._view_data.shape[0], self._max_vectors_thumbnail
                )
                vectors = copy(self._view_data[thumbnail_indices, :, -2:])
                thumbnail_color_indices = self._view_indices[thumbnail_indices]
            else:
                vectors = copy(self._view_data[:, :, -2:])
                thumbnail_color_indices = self._view_indices
            vectors[:, 1, :] = (
                vectors[:, 0, :] + vectors[:, 1, :] * self.length
            )
            downsampled = (vectors - offset) * zoom_factor
            downsampled = np.clip(
                downsampled, 0, np.subtract(self._thumbnail_shape[:2], 1)
            )
            edge_colors = self._edge.colors[thumbnail_color_indices]
            for v, ec in zip(downsampled, edge_colors, strict=False):
                start = v[0]
                stop = v[1]
                step = int(np.ceil(np.max(abs(stop - start))))
                x_vals = np.linspace(start[0], stop[0], step)
                y_vals = np.linspace(start[1], stop[1], step)
                for x, y in zip(x_vals, y_vals, strict=False):
                    colormapped[int(x), int(y), :] = ec
            colormapped[..., 3] = (colormapped[..., 3] * self.opacity).astype(
                np.uint8
            )
            self.thumbnail = colormapped

    def _get_value(self, position):
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : None
            Value of the data at the coord.
        """
        return
