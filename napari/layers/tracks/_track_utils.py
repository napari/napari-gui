import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from napari.layers.utils.layer_utils import _FeatureTable
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans


class TrackManager:
    """Manage track data and simplify interactions with the Tracks layer.

    Parameters
    ----------
    data : array
        See attribute doc below.

    Attributes
    ----------
    data : array (N, D+1)
        Coordinates for N points in D+1 dimensions. ID,T,(Z),Y,X. The first
        axis is the integer ID of the track. D is either 3 or 4 for planar
        or volumetric timeseries respectively.
    features : Dataframe-like
        Features table where each row corresponds to a point and each column
        is a feature.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    graph : dict {int: list}
        Graph representing associations between tracks. Dictionary defines the
        mapping between a track ID and the parents of the track. This can be
        one (the track has one parent, and the parent has >=1 child) in the
        case of track splitting, or more than one (the track has multiple
        parents, but only one child) in the case of track merging.
        See examples/tracks_3d_with_graph.py
    ndim : int
        Number of spatiotemporal dimensions of the data.
    max_time: float, int
        Maximum value of timestamps in data.
    track_vertices : array (N, D)
        Vertices for N points in D dimensions. T,(Z),Y,X
    track_connex : array (N,)
        Connection array specifying consecutive vertices that are linked to
        form the tracks. Boolean
    track_times : array (N,)
        Timestamp for each vertex in track_vertices.
    graph_vertices : array (N, D)
        Vertices for N points in D dimensions. T,(Z),Y,X
    graph_connex : array (N,)
        Connection array specifying consecutive vertices that are linked to
        form the graph.
    graph_times : array (N,)
        Timestamp for each vertex in graph_vertices.
    track_ids : array (N,)
        Track ID for each vertex in track_vertices.
    """

    def __init__(self, data: np.ndarray) -> None:
        # store the raw data here
        self.data = data
        self._feature_table = _FeatureTable()

        self._data: npt.NDArray
        self._order: list[int]
        self._kdtree: cKDTree
        self._points: npt.NDArray
        self._points_id: npt.NDArray
        self._points_lookup: dict[int, slice]
        self._ordered_points_idx: npt.NDArray

        self._track_vertices: npt.NDArray | None = None
        self._track_connex: npt.NDArray | None = None

        self._graph: dict[int, list[int]] | None = None
        self._graph_vertices = None
        self._graph_connex: npt.NDArray | None = None

    @staticmethod
    def _fast_points_lookup(sorted_time: np.ndarray) -> dict[int, slice]:
        """Computes a fast lookup table from time to their respective points slicing."""

        # finds where t transitions to t + 1
        transitions = np.nonzero(sorted_time[:-1] - sorted_time[1:])[0] + 1
        start = np.insert(transitions, 0, 0)

        # compute end of slice
        end = np.roll(start, -1)
        end[-1] = len(sorted_time)

        # access first position of each t slice
        time = sorted_time[start]

        return {
            t: slice(s, e) for s, e, t in zip(start, end, time, strict=False)
        }

    @property
    def data(self) -> np.ndarray:
        """array (N, D+1): Coordinates for N points in D+1 dimensions."""
        return self._data

    @data.setter
    def data(self, data: list | np.ndarray) -> None:
        """set the vertex data and build the vispy arrays for display"""

        # convert data to a numpy array if it is not already one
        data = np.asarray(data)

        # check check the formatting of the incoming track data
        data = self._validate_track_data(data)

        # Sort data by ID then time
        self._order = np.lexsort((data[:, 1], data[:, 0]))
        self._data = data[self._order]

        # build the indices for sorting points by time
        self._ordered_points_idx = np.argsort(self._data[:, 1])
        self._points = self._data[self._ordered_points_idx, 1:]

        # build a tree of the track data to allow fast lookup of nearest track
        self._kdtree = cKDTree(self._points)

        # make the lookup table
        # NOTE(arl): it's important to convert the time index to an integer
        # here to make sure that we align with the napari dims index which
        # will be an integer - however, the time index does not necessarily
        # need to be an int, and the shader will render correctly.
        time = np.round(self._points[:, 0]).astype(np.uint)
        self._points_lookup = self._fast_points_lookup(time)

        # make a second lookup table using a sparse matrix to convert track id
        # to the vertex indices
        self._id2idxs = coo_matrix(
            (
                np.broadcast_to(1, self.track_ids.size),  # just dummy ones
                (self.track_ids, np.arange(self.track_ids.size)),
            )
        ).tocsr()

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
        return self._feature_table.values

    @features.setter
    def features(
        self,
        features: dict[str, np.ndarray] | pd.DataFrame,
    ) -> None:
        self._feature_table.set_values(features, num_data=len(self.data))
        self._feature_table.reorder(self._order)
        if 'track_id' not in self._feature_table.values:
            self._feature_table.values['track_id'] = self.track_ids

    @property
    def properties(self) -> dict[str, np.ndarray]:
        """dict {str: np.ndarray (N,)}: Properties for each track."""
        return self._feature_table.properties()

    @properties.setter
    def properties(self, properties: dict[str, Array]) -> None:
        """set track properties"""
        self.features = properties

    @property
    def graph(self) -> dict[int, list[int]] | None:
        """dict {int: list}: Graph representing associations between tracks."""
        return self._graph

    @graph.setter
    def graph(self, graph: dict[int, int | list[int]]) -> None:
        """set the track graph"""
        self._graph = self._normalize_track_graph(graph)

    @property
    def track_ids(self) -> npt.NDArray[np.uint32]:
        """return the track identifiers"""
        return self.data[:, 0].astype(np.uint32)

    @property
    def unique_track_ids(self) -> npt.NDArray[np.uint32]:
        """return the unique track identifiers"""
        return np.unique(self.track_ids)

    def __len__(self) -> int:
        """return the number of tracks"""
        return len(self.unique_track_ids) if self.data is not None else 0

    def _vertex_indices_from_id(self, track_id: int) -> npt.NDArray:
        """return the vertices corresponding to a track id"""
        return self._id2idxs[track_id].nonzero()[1]

    def _validate_track_data(self, data: np.ndarray) -> np.ndarray:
        """validate the coordinate data"""

        if data.ndim != 2:
            raise ValueError(
                trans._('track vertices should be a NxD array', deferred=True)
            )

        if data.shape[1] < 4 or data.shape[1] > 5:
            raise ValueError(
                trans._(
                    'track vertices should be 4 or 5-dimensional',
                    deferred=True,
                )
            )

        # check that all IDs are integers
        ids = data[:, 0]
        if not np.array_equal(np.floor(ids), ids):
            raise ValueError(
                trans._('track id must be an integer', deferred=True)
            )

        if not all(t >= 0 for t in data[:, 1]):
            raise ValueError(
                trans._(
                    'track timestamps must be greater than zero', deferred=True
                )
            )

        return data

    def _normalize_track_graph(
        self, graph: dict[int, int | list[int]]
    ) -> dict[int, list[int]]:
        """validate the track graph"""
        new_graph: dict[int, list[int]] = {}

        # check that graph nodes are of correct format
        for node_idx, parents_idx in graph.items():
            # make sure parents are always a list
            if isinstance(parents_idx, list):
                new_graph[node_idx] = parents_idx
            else:
                new_graph[node_idx] = [parents_idx]

        unique_track_ids = set(self.unique_track_ids)

        # check that graph nodes exist in the track id lookup
        for node_idx, parents_idx in new_graph.items():
            nodes = [node_idx, *parents_idx]
            for node in nodes:
                if node not in unique_track_ids:
                    raise ValueError(
                        trans._(
                            'graph node {node_idx} not found',
                            deferred=True,
                            node_idx=node_idx,
                        )
                    )

        return new_graph

    def build_tracks(self) -> None:
        """build the tracks"""

        # Track ids associated to all vertices, sorted by time
        points_id = self.data[:, 0][self._ordered_points_idx]
        # Coordinates of all vertices
        track_vertices = self.data[:, 1:]

        # Indices in the data array just before the track id changes
        indices_new_id = np.where(np.diff(self.data[:, 0]))[0]

        # Define track_connex as an array full of 'True', then set to 'False'
        # at the indices just before the track id changes
        track_connex = np.ones(self.data.shape[0], dtype=bool)
        track_connex[indices_new_id] = False
        # Add 'False' for the last entry too (end of the last track)
        track_connex[-1] = False

        self._points_id = points_id
        self._track_vertices = track_vertices
        self._track_connex = track_connex

    def build_graph(self) -> None:
        """build the track graph"""

        graph_vertices = []
        graph_connex = []

        assert self.graph is not None
        for node_idx, parents_idx in self.graph.items():
            # we join from the first observation of the node, to the last
            # observation of the parent
            node_start = self._vertex_indices_from_id(node_idx)[0]
            node = self.data[node_start, 1:]

            for parent_idx in parents_idx:
                parent_stop = self._vertex_indices_from_id(parent_idx)[-1]
                parent = self.data[parent_stop, 1:]

                graph_vertices.append([node, parent])
                graph_connex.append([True, False])

        # if there is a graph, store the vertices and connection arrays,
        # otherwise, clear the vertex arrays
        if graph_vertices:
            self._graph_vertices = np.concatenate(graph_vertices, axis=0)
            self._graph_connex = np.concatenate(graph_connex, axis=0)
        else:
            self._graph_vertices = None
            self._graph_connex = None

    def vertex_properties(self, color_by: str) -> np.ndarray:
        """return the properties of tracks by vertex"""

        if color_by not in self.properties:
            raise ValueError(
                trans._(
                    'Property {color_by} not found',
                    deferred=True,
                    color_by=color_by,
                )
            )

        return self.properties[color_by]

    def get_value(self, coords: npt.NDArray) -> npt.NDArray | None:
        """use a kd-tree to lookup the ID of the nearest tree"""
        if self._kdtree is None:
            return None

        # query can return indices to points that do not exist, trim that here
        # then prune to only those in the current frame/time
        # NOTE(arl): I don't like this!!!
        d, idx = self._kdtree.query(coords, k=10)
        idx = [i for i in idx if i >= 0 and i < self._points.shape[0]]
        pruned = [i for i in idx if self._points[i, 0] == coords[0]]

        # if we have found a point, return it
        if pruned and self._points_id is not None:
            return self._points_id[pruned[0]]
        return None  # return the track ID

    @property
    def ndim(self) -> int:
        """Determine number of spatiotemporal dimensions of the layer."""
        return self.data.shape[1] - 1

    @property
    def max_time(self) -> int | None:
        """Determine the maximum timestamp of the dataset"""
        if self.track_times is not None:
            return int(np.max(self.track_times))
        return None

    @property
    def track_vertices(self) -> np.ndarray | None:
        """return the track vertices"""
        return self._track_vertices

    @property
    def track_connex(self) -> np.ndarray | None:
        """vertex connections for drawing track lines"""
        return self._track_connex

    @property
    def graph_vertices(self) -> np.ndarray | None:
        """return the graph vertices"""
        return self._graph_vertices

    @property
    def graph_connex(self) -> npt.NDArray | None:
        """vertex connections for drawing the graph"""
        return self._graph_connex

    @property
    def track_times(self) -> np.ndarray | None:
        """time points associated with each track vertex"""
        if self.track_vertices is not None:
            return self.track_vertices[:, 0]
        return None

    @property
    def graph_times(self) -> np.ndarray | None:
        """time points associated with each graph vertex"""
        if self.graph_vertices is not None:
            return self.graph_vertices[:, 0]
        return None

    def track_labels(
        self, current_time: int
    ) -> tuple[None, None] | tuple[list[str], np.ndarray]:
        """return track labels at the current time"""
        if self._points_id is None:
            return None, None
        # this is the slice into the time ordered points array
        if current_time not in self._points_lookup:
            lbl = []
            pos = np.array([])
        else:
            lookup = self._points_lookup[current_time]
            pos = self._points[lookup, ...]
            lbl = [f'ID:{i}' for i in self._points_id[lookup]]

        return lbl, pos
