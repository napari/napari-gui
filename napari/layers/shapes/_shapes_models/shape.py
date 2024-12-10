from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from napari.layers.shapes._shapes_utils import (
    is_collinear,
    path_to_mask,
    poly_to_mask,
    triangulate_edge,
    triangulate_face,
)
from napari.utils.misc import argsort
from napari.utils.translations import trans


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
        shape_type: str = 'rectangle',
        edge_width: float = 1,
        z_index: int = 0,
        dims_order: Optional[list[int]] = None,
        ndisplay: Literal[2] = 2,
    ) -> None:
        self._dims_order = dims_order or list(range(2))
        self._ndisplay = ndisplay
        self.slice_key: Optional[npt.NDArray] = None

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

    @property
    @abstractmethod
    def data(self) -> npt.NDArray:
        # user writes own docstring
        raise NotImplementedError

    @data.setter
    @abstractmethod
    def data(self, data: npt.NDArray) -> None:
        raise NotImplementedError

    @abstractmethod
    def _update_displayed_data(self) -> None:
        raise NotImplementedError

    @property
    def ndisplay(self) -> Literal[2]:
        """int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: Literal[2]) -> None:
        if self.ndisplay == ndisplay:
            return
        self._ndisplay = ndisplay
        self._update_displayed_data()

    @property
    def dims_order(self) -> list[int]:
        """(D,) list: Order that the dimensions are rendered in."""
        return self._dims_order

    @dims_order.setter
    def dims_order(self, dims_order: list[int]) -> None:
        if self.dims_order == dims_order:
            return
        self._dims_order = dims_order
        self._update_displayed_data()

    @property
    def dims_displayed(self) -> list[int]:
        """list: Dimensions that are displayed."""
        return self.dims_order[-self.ndisplay :]

    def bounding_box(self) -> np.ndarray:
        """(2, N) array, bounding box of the object."""
        # We add +-0.5 to handle edge width
        return self._bounding_box[:, self.dims_displayed] + [
            [-0.5 * self.edge_width],
            [0.5 * self.edge_width],
        ]

    @property
    def dims_not_displayed(self) -> list[int]:
        """list: Dimensions that are not displayed."""
        return self.dims_order[: -self.ndisplay]

    @property
    def data_displayed(self) -> npt.NDArray:
        """(N, 2) array: Vertices of the shape that are currently displayed."""
        return self.data[:, self.dims_displayed]

    @property
    def edge_width(self) -> float:
        """float: thickness of lines and edges."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: float) -> None:
        self._edge_width = edge_width

    @property
    def z_index(self) -> int:
        """int: z order priority of shape. Shapes with higher z order displayed
        ontop of others.
        """
        return self._z_index

    @z_index.setter
    def z_index(self, z_index: int) -> None:
        self._z_index = z_index

    def _set_meshes(
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

    def _all_triangles(self) -> npt.NDArray:
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

        points = self.data_displayed

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

    def scale(
        self,
        scale: Union[float, list[float]],
        center: Optional[npt.NDArray] = None,
    ) -> None:
        """Performs a scaling on the shape

        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : np.ndarray
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

    def rotate(
        self, angle: float, center: Optional[npt.NDArray] = None
    ) -> None:
        """Performs a rotation on the shape

        Parameters
        ----------
        angle : float
            angle specifying rotation of shape in degrees. CCW is positive.
        center : np.ndarray
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

    def flip(
        self, axis: Literal[0, 1], center: Optional[npt.NDArray] = None
    ) -> None:
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

    def to_mask(
        self,
        mask_shape: Optional[npt.NDArray] = None,
        zoom_factor: float = 1,
        offset: tuple[int, int] = (0, 0),
    ) -> npt.NDArray:
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
