# from napari.layers.base.base import Layer
# from napari.utils.events import Event
# from napari.utils.colormaps import AVAILABLE_COLORMAPS

from typing import Any
from warnings import warn

import numpy as np
import pandas as pd

from napari.layers.base import Layer
from napari.layers.tracks._track_utils import TrackManager
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from napari.utils.events import Event
from napari.utils.translations import trans


class Tracks(Layer):
    """Tracks layer.

    Parameters
    ----------
    data : array (N, D+1)
        Coordinates for N points in D+1 dimensions. ID,T,(Z),Y,X. The first
        axis is the integer ID of the track. D is either 3 or 4 for planar
        or volumetric timeseries respectively.
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
    color_by : str
        Track property (from property keys) by which to color vertices.
    colormap : str
        Default colormap to use to set vertex colors. Specialized colormaps,
        relating to specified properties can be passed to the layer via
        colormaps_dict.
    colormaps_dict : dict {str: napari.utils.Colormap}
        Optional dictionary mapping each property to a colormap for that
        property. This allows each property to be assigned a specific colormap,
        rather than having a global colormap for everything.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    features : Dataframe-like
        Features table where each row corresponds to a point and each column
        is a feature.
    graph : dict {int: list}
        Graph representing associations between tracks. Dictionary defines the
        mapping between a track ID and the parents of the track. This can be
        one (the track has one parent, and the parent has >=1 child) in the
        case of track splitting, or more than one (the track has multiple
        parents, but only one child) in the case of track merging.
        See examples/tracks_3d_with_graph.py
    head_length : float
        Length of the positive (forward in time) tails in units of time.
    metadata : dict
        Layer metadata.
    name : str
        Name of the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    projection_mode : str
        How data outside the viewed dimensions but inside the thick Dims slice will
        be projected onto the viewed dimenions.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
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
    tail_length : float
        Length of the positive (backward in time) tails in units of time.
    tail_width : float
        Width of the track tails in pixels.
    translate : tuple of float
        Translation values for the layer.
    units : tuple of str or pint.Unit, optional
        Units of the layer data in world coordinates.
        If not provided, the default units are assumed to be pixels.
    visible : bool
        Whether the layer visual is currently being displayed.
    """

    # The max number of tracks that will ever be used to render the thumbnail
    # If more tracks are present then they are randomly subsampled
    _max_tracks_thumbnail = 1024

    def __init__(
        self,
        data,
        *,
        affine=None,
        axis_labels=None,
        blending='additive',
        cache=True,
        color_by='track_id',
        colormap='turbo',
        colormaps_dict=None,
        experimental_clipping_planes=None,
        features=None,
        graph=None,
        head_length: int = 0,
        metadata=None,
        name=None,
        opacity=1.0,
        projection_mode='none',
        properties=None,
        rotate=None,
        scale=None,
        shear=None,
        tail_length: int = 30,
        tail_width: int = 2,
        translate=None,
        units=None,
        visible=True,
    ) -> None:
        # if not provided with any data, set up an empty layer in 2D+t
        # otherwise convert the data to an np.ndarray
        data = np.empty((0, 4)) if data is None else np.asarray(data)

        # set the track data dimensions (remove ID from data)
        ndim = data.shape[1] - 1

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

        self.events.add(
            tail_width=Event,
            tail_length=Event,
            head_length=Event,
            display_id=Event,
            display_tail=Event,
            display_graph=Event,
            color_by=Event,
            colormap=Event,
            properties=Event,
            rebuild_tracks=Event,
            rebuild_graph=Event,
        )

        # track manager deals with data slicing, graph building and properties
        self._manager = TrackManager(data)

        self._track_colors: np.ndarray | None = None
        self._colormaps_dict = colormaps_dict or {}  # additional colormaps
        self._color_by = color_by  # default color by ID
        self._colormap = colormap

        # use this to update shaders when the displayed dims change
        self._current_displayed_dims = None

        # track display default limits
        self._max_length = 300
        self._max_width = 20

        # track display properties
        self.tail_width = tail_width
        self.tail_length = tail_length
        self.head_length = head_length
        self.display_id = False
        self.display_tail = True
        self.display_graph = True

        # set the data, features, and graph
        self.data = data
        if properties is not None:
            self.properties = properties
        else:
            self.features = features
        self.graph = graph or {}

        self.color_by = color_by
        self.colormap = colormap

        self.refresh()

        # reset the display before returning
        self._current_displayed_dims = None

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
            maxs = np.max(self.data, axis=0)
            mins = np.min(self.data, axis=0)
            extrema = np.vstack([mins, maxs])
        return extrema[:, 1:]

    def _get_ndim(self) -> int:
        """Determine number of dimensions of the layer."""
        return self._manager.ndim

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
                'data': self.data,
                'properties': self.properties,
                'graph': self.graph,
                'color_by': self.color_by,
                'colormap': self.colormap,
                'colormaps_dict': self.colormaps_dict,
                'tail_width': self.tail_width,
                'tail_length': self.tail_length,
                'head_length': self.head_length,
                'features': self.features,
            }
        )
        return state

    def _set_view_slice(self) -> None:
        """Sets the view given the indices to slice with."""

        # if the displayed dims have changed, update the shader data
        dims_displayed = self._slice_input.displayed
        if dims_displayed != self._current_displayed_dims:
            # store the new dims
            self._current_displayed_dims = dims_displayed
            # fire the events to update the shaders
            self.events.rebuild_tracks()
            self.events.rebuild_graph()

        return

    def _get_value(self, position) -> int | None:
        """Value of the data at a position in data coordinates.

        Use a kd-tree to lookup the ID of the nearest tree.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : int or None
            Index of track that is at the current coordinate if any.
        """
        val = self._manager.get_value(np.array(position))
        if val is None:
            return None
        return int(val)

    def _update_thumbnail(self) -> None:
        """Update thumbnail with current points and colors."""
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1

        if self._view_data is not None and self.track_colors is not None:
            de = self._extent_data
            min_vals = [de[0, i] for i in self._slice_input.displayed]
            shape = np.ceil(
                [de[1, i] - de[0, i] + 1 for i in self._slice_input.displayed]
            ).astype(int)
            zoom_factor = np.divide(
                self._thumbnail_shape[:2], shape[-2:]
            ).min()
            if len(self._view_data) > self._max_tracks_thumbnail:
                thumbnail_indices = np.random.randint(
                    0, len(self._view_data), self._max_tracks_thumbnail
                )
                points = self._view_data[thumbnail_indices]
            else:
                points = self._view_data
                thumbnail_indices = np.array(range(len(self._view_data)))

            # get the track coords here
            coords = np.floor(
                (points[:, :2] - min_vals[1:] + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(
                coords, 0, np.subtract(self._thumbnail_shape[:2], 1)
            )

            # modulate track colors as per colormap/current_time
            assert self.track_times is not None
            assert self.current_time is not None
            colors = self.track_colors[thumbnail_indices]
            times = self.track_times[thumbnail_indices]
            alpha = (self.head_length + self.current_time - times) / (
                self.tail_length + self.head_length
            )
            alpha[times > self.current_time] = 1.0
            colors[:, -1] = np.clip(1.0 - alpha, 0.0, 1.0)
            colormapped[coords[:, 1], coords[:, 0]] = colors

        colormapped[..., 3] *= self.opacity
        colormapped[np.isnan(colormapped)] = 0
        self.thumbnail = colormapped.astype(np.uint8)

    @property
    def _view_data(self):
        """return a view of the data"""
        return self._pad_display_data(self._manager.track_vertices)

    @property
    def _view_graph(self):
        """return a view of the graph"""
        return self._pad_display_data(self._manager.graph_vertices)

    def _pad_display_data(self, vertices):
        """pad display data when moving between 2d and 3d"""
        if vertices is None:
            return None

        data = vertices[:, self._slice_input.displayed]
        # if we're only displaying two dimensions, then pad the display dim
        # with zeros
        if self._slice_input.ndisplay == 2:
            data = np.pad(data, ((0, 0), (0, 1)), 'constant')
            return data[:, (1, 0, 2)]  # y, x, z -> x, y, z

        return data[:, (2, 1, 0)]  # z, y, x -> x, y, z

    @property
    def current_time(self) -> int | None:
        """current time according to the first dimension"""
        # TODO(arl): get the correct index here
        time_step = self._data_slice.point[0]

        if isinstance(time_step, slice):
            # if we are visualizing all time, then just set to the maximum
            # timestamp of the dataset
            return self._manager.max_time

        return time_step

    @property
    def use_fade(self) -> bool:
        """toggle whether we fade the tail of the track, depending on whether
        the time dimension is displayed"""
        return 0 in self._slice_input.not_displayed

    @property
    def data(self) -> np.ndarray:
        """array (N, D+1): Coordinates for N points in D+1 dimensions."""
        return self._manager.data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """set the data and build the vispy arrays for display"""
        # set the data and build the tracks
        self._manager.data = data
        self._manager.build_tracks()

        # reset the properties and recolor the tracks
        self.features = {}
        self._recolor_tracks()

        # reset the graph
        self._manager.graph = {}
        self._manager.build_graph()

        # fire events to update shaders
        self._update_dims()
        self.events.rebuild_tracks()
        self.events.rebuild_graph()
        self.events.data(value=self.data)
        self._reset_editable()

    @property
    def features(self) -> pd.DataFrame:
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
        return self._manager.features

    @features.setter
    def features(
        self,
        features: dict[str, np.ndarray] | pd.DataFrame,
    ) -> None:
        self._manager.features = features
        self._check_color_by_in_features()
        self.events.properties()

    @property
    def properties(self) -> dict[str, np.ndarray]:
        """dict {str: np.ndarray (N,)}: Properties for each track."""
        return self._manager.properties

    @properties.setter
    def properties(self, properties: dict[str, np.ndarray]) -> None:
        """set track properties"""
        self.features = properties

    @property
    def properties_to_color_by(self) -> list[str]:
        """track properties that can be used for coloring etc..."""
        return list(self.properties.keys())

    @property
    def graph(self) -> dict[int, list[int]] | None:
        """dict {int: list}: Graph representing associations between tracks."""
        return self._manager.graph

    @graph.setter
    def graph(self, graph: dict[int, int | list[int]]) -> None:
        """Set the track graph."""
        # Ignored type, because mypy can't handle different signatures
        # on getters and setters; see https://github.com/python/mypy/issues/3004
        self._manager.graph = graph  # type: ignore[assignment]
        self._manager.build_graph()
        self.events.rebuild_graph()

    @property
    def tail_width(self) -> float:
        """float: Width for all vectors in pixels."""
        return self._tail_width

    @tail_width.setter
    def tail_width(self, tail_width: float) -> None:
        self._tail_width: float = np.clip(tail_width, 0.5, self._max_width)
        self.events.tail_width()

    @property
    def tail_length(self) -> int:
        """float: Width for all vectors in pixels."""
        return self._tail_length

    @tail_length.setter
    def tail_length(self, tail_length: int) -> None:
        if tail_length > self._max_length:
            self._max_length = tail_length
        self._tail_length: int = tail_length
        self.events.tail_length()

    @property
    def head_length(self) -> int:
        return self._head_length

    @head_length.setter
    def head_length(self, head_length: int) -> None:
        if head_length > self._max_length:
            self._max_length = head_length
        self._head_length: int = head_length
        self.events.head_length()

    @property
    def display_id(self) -> bool:
        """display the track id"""
        return self._display_id

    @display_id.setter
    def display_id(self, value: bool) -> None:
        self._display_id = value
        self.events.display_id()
        # TODO: this refresh is only here to trigger setting the id text...
        #       a bit overkill? But maybe for a future PR.
        self.refresh(extent=False, thumbnail=False)

    @property
    def display_tail(self) -> bool:
        """display the track tail"""
        return self._display_tail

    @display_tail.setter
    def display_tail(self, value: bool) -> None:
        self._display_tail = value
        self.events.display_tail()

    @property
    def display_graph(self) -> bool:
        """display the graph edges"""
        return self._display_graph

    @display_graph.setter
    def display_graph(self, value: bool) -> None:
        self._display_graph = value
        self.events.display_graph()

    @property
    def color_by(self) -> str:
        return self._color_by

    @color_by.setter
    def color_by(self, color_by: str) -> None:
        """set the property to color vertices by"""
        if color_by not in self.properties_to_color_by:
            raise ValueError(
                trans._(
                    '{color_by} is not a valid property key',
                    deferred=True,
                    color_by=color_by,
                )
            )
        self._color_by = color_by
        self._recolor_tracks()
        self.events.color_by()

    @property
    def colormap(self) -> str:
        return self._colormap

    @colormap.setter
    def colormap(self, colormap: str) -> None:
        """set the default colormap"""
        if colormap not in AVAILABLE_COLORMAPS:
            raise ValueError(
                trans._(
                    'Colormap {colormap} not available',
                    deferred=True,
                    colormap=colormap,
                )
            )
        self._colormap = colormap
        self._recolor_tracks()
        self.events.colormap()

    @property
    def colormaps_dict(self) -> dict[str, Colormap]:
        return self._colormaps_dict

    # Ignored type because mypy doesn't recognise colormaps_dict as a property
    # TODO: investigate and fix this - not sure why this is the case?
    @colormaps_dict.setter  # type: ignore[attr-defined]
    def colomaps_dict(self, colormaps_dict: dict[str, Colormap]) -> None:
        # validate the dictionary entries?
        self._colormaps_dict = colormaps_dict

    def _recolor_tracks(self) -> None:
        """recolor the tracks"""

        # this catch prevents a problem coloring the tracks if the data is
        # updated before the properties are. properties should always contain
        # a track_id key
        if self.color_by not in self.properties_to_color_by:
            self._color_by = 'track_id'
            self.events.color_by()

        # if we change the coloring, rebuild the vertex colors array
        vertex_properties = self._manager.vertex_properties(self.color_by)

        def _norm(p):
            return (p - np.min(p)) / np.max([1e-10, np.ptp(p)])

        if self.color_by in self.colormaps_dict:
            colormap = self.colormaps_dict[self.color_by]
        else:
            # if we don't have a colormap, get one and scale the properties
            colormap = AVAILABLE_COLORMAPS[self.colormap]
            vertex_properties = _norm(vertex_properties)

        # actually set the vertex colors
        self._track_colors = colormap.map(vertex_properties)

    @property
    def track_connex(self) -> np.ndarray | None:
        """vertex connections for drawing track lines"""
        return self._manager.track_connex

    @property
    def track_colors(self) -> np.ndarray | None:
        """return the vertex colors according to the currently selected
        property"""
        return self._track_colors

    @property
    def graph_connex(self) -> np.ndarray | None:
        """vertex connections for drawing the graph"""
        return self._manager.graph_connex

    @property
    def track_times(self) -> np.ndarray | None:
        """time points associated with each track vertex"""
        return self._manager.track_times

    @property
    def graph_times(self) -> np.ndarray | None:
        """time points associated with each graph vertex"""
        return self._manager.graph_times

    @property
    def track_labels(self) -> tuple:
        """return track labels at the current time"""
        assert self.current_time is not None
        labels, positions = self._manager.track_labels(self.current_time)

        # if there are no labels, return empty for vispy
        if not labels:
            return None, (None, None)

        padded_positions = self._pad_display_data(positions)
        return labels, padded_positions

    def _check_color_by_in_features(self) -> None:
        if self._color_by not in self.features.columns:
            warn(
                (
                    trans._(
                        'Previous color_by key {key!r} not present in features. Falling back to track_id',
                        deferred=True,
                        key=self._color_by,
                    )
                ),
                UserWarning,
            )
            self._color_by = 'track_id'
            self.events.color_by()
