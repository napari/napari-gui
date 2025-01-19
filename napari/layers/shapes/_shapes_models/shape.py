from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import numpy.typing as npt

from napari.layers.shapes._shapes_utils import (
    is_collinear,
    path_to_mask,
    poly_to_mask,
    triangulate_edge,
    triangulate_face,
)
from napari.settings import get_settings
from napari.utils.misc import argsort
from napari.utils.translations import trans

try:
    from PartSegCore_compiled_backend.triangulate import (
        triangulate_path_edge_numpy,
        triangulate_polygon_numpy_li,
        triangulate_polygon_with_edge_numpy_li,
    )

except ImportError:
    triangulate_path_edge_numpy = None
    triangulate_polygon_numpy_li = None
    triangulate_polygon_with_edge_numpy_li = None


def _remove_path_duplicates_np(data: np.ndarray, closed: bool):
    # We add the first data point at the end to get the same length bool
    # array as the data, and also to work on closed shapes; the last value
    # in the diff array compares the last and first vertex.
    diff = np.diff(np.append(data, data[0:1, :], axis=0), axis=0)
    dup = np.all(diff == 0, axis=1)
    # if the shape is closed, check whether the first vertex is the same
    # as the last vertex, and count it as a duplicate if so
    if closed and dup[-1]:
        dup[0] = True
    # we allow repeated nodes at the end for the lasso tool, which
    # for an instant needs both the last placed point and the point at the
    # cursor to be the same; if the lasso implementation becomes cleaner,
    # remove this hardcoding
    dup[-2:] = False
    indices = np.arange(data.shape[0])
    return data[indices[~dup]]


try:
    from napari.layers.shapes._accelerated_triangulate import (
        remove_path_duplicates,
    )
except ImportError:
    remove_path_duplicates = _remove_path_duplicates_np


class Shape(ABC):
    """Base class for a single shape

    Parameters
    ----------
    data : (N, D) array
        Vertices specifying the shape.
    edge_width : float
        thickness of lines and edges.
    z_index : int
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others.
    dims_order : (D,) list
        Order that the dimensions are to be rendered in.
    ndisplay : int
        Number of displayed dimensions.

    Attributes
    ----------
    data : (N, D) array
        Vertices specifying the shape.
    data_displayed : (N, 2) array
        Vertices of the shape that are currently displayed. Only 2D rendering
        currently supported.
    edge_width : float
        thickness of lines and edges.
    name : str
        Name of shape type.
    z_index : int
        Specifier of z order priority. Shapes with higher z order are displayed
        ontop of others.
    dims_order : (D,) list
        Order that the dimensions are rendered in.
    ndisplay : int
        Number of dimensions to be displayed, must be 2 as only 2D rendering
        currently supported.
    displayed : tuple
        List of dimensions that are displayed.
    not_displayed : tuple
        List of dimensions that are not displayed.
    slice_key : (2, M) array
        Min and max values of the M non-displayed dimensions, useful for
        slicing multidimensional shapes.

    Notes
    -----
    _closed : bool
        Bool if shape edge is a closed path or not
    _box : np.ndarray
        9x2 array of vertices of the interaction box. The first 8 points are
        the corners and midpoints of the box in clockwise order starting in the
        upper-left corner. The last point is the center of the box
    _face_vertices : np.ndarray
        Qx2 array of vertices of all triangles for the shape face
    _face_triangles : np.ndarray
        Px3 array of vertex indices that form the triangles for the shape face
    _edge_vertices : np.ndarray
        Rx2 array of centers of vertices of triangles for the shape edge.
        These values should be added to the scaled `_edge_offsets` to get the
        actual vertex positions. The scaling corresponds to the width of the
        edge
    _edge_offsets : np.ndarray
        Sx2 array of offsets of vertices of triangles for the shape edge. For
        These values should be scaled and added to the `_edge_vertices` to get
        the actual vertex positions. The scaling corresponds to the width of
        the edge
    _edge_triangles : np.ndarray
        Tx3 array of vertex indices that form the triangles for the shape edge
    _filled : bool
        Flag if array is filled or not.
    _use_face_vertices : bool
        Flag to use face vertices for mask generation.
    """

    def __init__(
        self,
        *,
        shape_type='rectangle',
        edge_width=1,
        z_index=0,
        dims_order=None,
        ndisplay=2,
    ) -> None:
        self._dims_order = dims_order or list(range(2))
        self._ndisplay = ndisplay
        self.slice_key: npt.NDArray

        self._face_vertices = np.empty((0, self.ndisplay))
        self._face_triangles = np.empty((0, 3), dtype=np.uint32)
        self._edge_vertices = np.empty((0, self.ndisplay))
        self._edge_offsets = np.empty((0, self.ndisplay))
        self._edge_triangles = np.empty((0, 3), dtype=np.uint32)
        self._box = np.empty((9, 2))

        self._closed = False
        self._filled = True
        self._use_face_vertices = False
        self.edge_width = edge_width
        self.z_index = z_index
        self.name = ''

        self._data: npt.NDArray
        self._bounding_box = np.empty((0, self.ndisplay))

    def __new__(cls, *args, **kwargs):
        if (
            get_settings().experimental.compiled_triangulation
            and triangulate_path_edge_numpy is not None
        ):
            cls._set_meshes = cls._set_meshes_compiled
        else:
            cls._set_meshes = cls._set_meshes_py
        return super().__new__(cls)

    @property
    @abstractmethod
    def data(self):
        # user writes own docstring
        raise NotImplementedError

    @data.setter
    @abstractmethod
    def data(self, data):
        raise NotImplementedError

    @abstractmethod
    def _update_displayed_data(self) -> None:
        raise NotImplementedError

    @property
    def ndisplay(self):
        """int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        if self.ndisplay == ndisplay:
            return
        self._ndisplay = ndisplay
        self._update_displayed_data()

    @property
    def dims_order(self):
        """(D,) list: Order that the dimensions are rendered in."""
        return self._dims_order

    @dims_order.setter
    def dims_order(self, dims_order):
        if self.dims_order == dims_order:
            return
        self._dims_order = dims_order
        self._update_displayed_data()

    @cached_property
    def dims_displayed(self):
        """tuple: Dimensions that are displayed."""
        return self.dims_order[-self.ndisplay :]

    @property
    def bounding_box(self) -> np.ndarray:
        """(2, N) array, bounding box of the object."""
        # We add +-0.5 to handle edge width
        return self._bounding_box[:, self.dims_displayed] + [
            [-0.5 * self.edge_width],
            [0.5 * self.edge_width],
        ]

    @property
    def dims_not_displayed(self):
        """tuple: Dimensions that are not displayed."""
        return self.dims_order[: -self.ndisplay]

    @cached_property
    def data_displayed(self):
        """(N, 2) array: Vertices of the shape that are currently displayed."""
        return self.data[:, self.dims_displayed]

    @property
    def edge_width(self):
        """float: thickness of lines and edges."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width):
        self._edge_width = edge_width

    @property
    def z_index(self):
        """int: z order priority of shape. Shapes with higher z order displayed
        ontop of others.
        """
        return self._z_index

    @z_index.setter
    def z_index(self, z_index):
        self._z_index = z_index

    def _set_meshes_compiled(
        self,
        data: npt.NDArray,
        closed: bool = True,
        face: bool = True,
        edge: bool = True,
    ) -> None:
        """Sets the face and edge meshes from a set of points.

        Parameters
        ----------
        data : np.ndarray
            Nx2 or Nx3 array specifying the shape to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not
        face : bool
            Bool which determines if the face need to be traingulated
        edge : bool
            Bool which determines if the edge need to be traingulated
        """
        if data.shape[1] == 3:
            self._set_meshes_py(data, closed=closed, face=face, edge=edge)
            return

        # if we are computing both edge and face triangles, we can do so
        # with a single call to the compiled backend
        if edge and face:
            (triangles, vertices), (centers, offsets, edge_triangles) = (
                triangulate_polygon_with_edge_numpy_li([data])
            )
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = edge_triangles
            self._face_vertices = vertices
            self._face_triangles = triangles
            return

        # otherwise, we make individual calls to specialized functions
        if edge:
            centers, offsets, triangles = triangulate_path_edge_numpy(
                data, closed=closed
            )
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
        else:
            self._edge_vertices = np.empty((0, self.ndisplay))
            self._edge_offsets = np.empty((0, self.ndisplay))
            self._edge_triangles = np.empty((0, 3), dtype=np.uint32)
        if face:
            triangles, vertices = triangulate_polygon_numpy_li([data])
            self._face_vertices = vertices
            self._face_triangles = triangles
        else:
            self._face_vertices = np.empty((0, self.ndisplay))
            self._face_triangles = np.empty((0, 3), dtype=np.uint32)

    def _set_meshes(  # noqa: B027
        self,
        data: npt.NDArray,
        closed: bool = True,
        face: bool = True,
        edge: bool = True,
    ) -> None: ...

    def _set_meshes_py(
        self,
        data: npt.NDArray,
        closed: bool = True,
        face: bool = True,
        edge: bool = True,
    ) -> None:
        """Sets the face and edge meshes from a set of points.

        Parameters
        ----------
        data : np.ndarray
            Nx2 or Nx3 array specifying the shape to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not
        face : bool
            Bool which determines if the face need to be traingulated
        edge : bool
            Bool which determines if the edge need to be traingulated
        """
        data = remove_path_duplicates(data, closed=closed)
        if edge:
            centers, offsets, triangles = triangulate_edge(data, closed=closed)
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
        else:
            self._edge_vertices = np.empty((0, self.ndisplay))
            self._edge_offsets = np.empty((0, self.ndisplay))
            self._edge_triangles = np.empty((0, 3), dtype=np.uint32)

        if face:
            idx = np.concatenate(
                [[True], ~np.all(data[1:] == data[:-1], axis=-1)]
            )
            clean_data = data[idx].copy()

            if not is_collinear(clean_data[:, -2:]):
                if clean_data.shape[1] == 2:
                    vertices, triangles = triangulate_face(clean_data)
                elif len(np.unique(clean_data[:, 0])) == 1:
                    val = np.unique(clean_data[:, 0])
                    vertices, triangles = triangulate_face(clean_data[:, -2:])
                    exp = np.expand_dims(np.repeat(val, len(vertices)), axis=1)
                    vertices = np.concatenate([exp, vertices], axis=1)
                else:
                    triangles = np.array([])
                    vertices = np.array([])
                if len(triangles) > 0:
                    self._face_vertices = vertices
                    self._face_triangles = triangles
                else:
                    self._face_vertices = np.empty((0, self.ndisplay))
                    self._face_triangles = np.empty((0, 3), dtype=np.uint32)
            else:
                self._face_vertices = np.empty((0, self.ndisplay))
                self._face_triangles = np.empty((0, 3), dtype=np.uint32)
        else:
            self._face_vertices = np.empty((0, self.ndisplay))
            self._face_triangles = np.empty((0, 3), dtype=np.uint32)

    def _all_triangles(self):
        """Return all triangles for the shape

        Returns
        -------
        np.ndarray
            Nx3 array of vertex indices that form the triangles for the shape
        """
        return np.vstack(
            [
                self._face_vertices[self._face_triangles],
                (self._edge_vertices + self.edge_width * self._edge_offsets)[
                    self._edge_triangles
                ],
            ]
        )

    def transform(self, transform: npt.NDArray) -> None:
        """Performs a linear transform on the shape

        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        self._box = self._box @ transform.T
        self._data[:, self.dims_displayed] = (
            self._data[:, self.dims_displayed] @ transform.T
        )
        self._face_vertices = self._face_vertices @ transform.T
        self.__dict__.pop('data_displayed', None)  # clear cache
        points = self.data_displayed
        points = remove_path_duplicates(points, closed=self._closed)
        centers, offsets, triangles = triangulate_edge(
            points, closed=self._closed
        )
        self._edge_vertices = centers
        self._edge_offsets = offsets
        self._edge_triangles = triangles
        self._bounding_box = np.array(
            [
                np.min(self._data, axis=0),
                np.max(self._data, axis=0),
            ]
        )
        self._clean_cache()

    def shift(self, shift: npt.NDArray) -> None:
        """Performs a 2D shift on the shape

        Parameters
        ----------
        shift : np.ndarray
            length 2 array specifying shift of shapes.
        """
        shift = np.array(shift)

        self._face_vertices = self._face_vertices + shift
        self._edge_vertices = self._edge_vertices + shift
        self._box = self._box + shift
        self._data[:, self.dims_displayed] = self.data_displayed + shift
        self._bounding_box[:, self.dims_displayed] = (
            self._bounding_box[:, self.dims_displayed] + shift
        )
        self._clean_cache()

    def scale(self, scale, center=None):
        """Performs a scaling on the shape

        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : list
            length 2 list specifying coordinate of center of scaling.
        """
        if isinstance(scale, (list, np.ndarray)):
            transform = np.array([[scale[0], 0], [0, scale[1]]])
        else:
            transform = np.array([[scale, 0], [0, scale]])
        if center is None:
            self.transform(transform)
        else:
            center = np.array(center)
            self.shift(-center)
            self.transform(transform)
            self.shift(center)

    def rotate(self, angle, center=None):
        """Performs a rotation on the shape

        Parameters
        ----------
        angle : float
            angle specifying rotation of shape in degrees. CCW is positive.
        center : list
            length 2 list specifying coordinate of fixed point of the rotation.
        """
        theta = np.radians(angle)
        transform = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        if center is None:
            self.transform(transform)
        else:
            center = np.array(center)
            self.shift(-center)
            self.transform(transform)
            self.shift(center)

    def flip(self, axis, center=None):
        """Performs a flip on the shape, either horizontal or vertical.

        Parameters
        ----------
        axis : int
            integer specifying axis of flip. `0` flips horizontal, `1` flips
            vertical.
        center : list
            length 2 list specifying coordinate of center of flip axes.
        """
        if axis == 0:
            transform = np.array([[1, 0], [0, -1]])
        elif axis == 1:
            transform = np.array([[-1, 0], [0, 1]])
        else:
            raise ValueError(
                trans._(
                    'Axis not recognized, must be one of "{{0, 1}}"',
                    deferred=True,
                )
            )
        if center is None:
            self.transform(transform)
        else:
            self.shift(-center)
            self.transform(transform)
            self.shift(-center)

    def to_mask(self, mask_shape=None, zoom_factor=1, offset=(0, 0)):
        """Convert the shape vertices to a boolean mask.

        Set points to `True` if they are lying inside the shape if the shape is
        filled, or if they are lying along the boundary of the shape if the
        shape is not filled. Negative points or points outside the mask_shape
        after the zoom and offset are clipped.

        Parameters
        ----------
        mask_shape : (D,) array
            Shape of mask to be generated. If non specified, takes the max of
            the displayed vertices.
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.

        Returns
        -------
        mask : np.ndarray
            Boolean array with `True` for points inside the shape
        """
        if mask_shape is None:
            mask_shape = np.round(self.data_displayed.max(axis=0)).astype(
                'int'
            )

        if len(mask_shape) == 2:
            embedded = False
            shape_plane = mask_shape
        elif len(mask_shape) == self.data.shape[1]:
            embedded = True
            shape_plane = [mask_shape[d] for d in self.dims_displayed]
        else:
            raise ValueError(
                trans._(
                    'mask shape length must either be 2 or the same as the dimensionality of the shape, expected {expected} got {received}.',
                    deferred=True,
                    expected=self.data.shape[1],
                    received=len(mask_shape),
                )
            )

        if self._use_face_vertices:
            data = self._face_vertices
        else:
            data = self.data_displayed

        data = data[:, -len(shape_plane) :]

        if self._filled:
            mask_p = poly_to_mask(shape_plane, (data - offset) * zoom_factor)
        else:
            mask_p = path_to_mask(shape_plane, (data - offset) * zoom_factor)

        # If the mask is to be embedded in a larger array, compute array
        # and embed as a slice.
        if embedded:
            mask = np.zeros(mask_shape, dtype=bool)
            slice_key: list[int | slice] = [0] * len(mask_shape)
            for i in range(len(mask_shape)):
                if i in self.dims_displayed:
                    slice_key[i] = slice(None)
                elif self.slice_key is not None:
                    slice_key[i] = slice(
                        self.slice_key[0, i], self.slice_key[1, i] + 1
                    )
                else:
                    raise RuntimeError(
                        'Internal error: self.slice_key is None'
                    )
            displayed_order = argsort(self.dims_displayed)
            mask[tuple(slice_key)] = mask_p.transpose(displayed_order)
        else:
            mask = mask_p

        return mask

    def _clean_cache(self) -> None:
        if 'dims_displayed' in self.__dict__:
            del self.__dict__['dims_displayed']
        if 'data_displayed' in self.__dict__:
            del self.__dict__['data_displayed']
