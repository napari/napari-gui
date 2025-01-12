import typing
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from functools import cached_property, wraps
from typing import Literal

import numpy as np
import numpy.typing as npt

from napari.layers.shapes._mesh import Mesh
from napari.layers.shapes._shapes_constants import ShapeType, shape_classes
from napari.layers.shapes._shapes_models import Line, Path, Shape
from napari.layers.shapes._shapes_utils import triangles_intersect_box
from napari.utils.geometry import (
    inside_triangles,
    intersect_line_with_triangles,
    line_in_triangles_3d,
)
from napari.utils.translations import trans


def _batch_dec(meth):
    """
    Decorator to apply `self.batched_updates` to the current method.
    """

    @wraps(meth)
    def _wrapped(self, *args, **kwargs):
        with self.batched_updates():
            return meth(self, *args, **kwargs)

    return _wrapped


class ShapeList:
    """List of shapes class.

    Parameters
    ----------
    data : list
        List of Shape objects
    ndisplay : int
        Number of displayed dimensions.

    Attributes
    ----------
    shapes : (N, ) list
        Shape objects.
    data : (N, ) list of (M, D) array
        Data arrays for each shape.
    ndisplay : int
        Number of displayed dimensions.
    slice_keys : (N, 2, P) array
        Array of slice keys for each shape. Each slice key has the min and max
        values of the P non-displayed dimensions, useful for slicing
        multidimensional shapes. If the both min and max values of shape are
        equal then the shape is entirely contained within the slice specified
        by those values.
    shape_types : (N, ) list of str
        Name of shape type for each shape.
    edge_color : (N x 4) np.ndarray
        Array of RGBA edge colors for each shape.
    face_color : (N x 4) np.ndarray
        Array of RGBA face colors for each shape.
    edge_widths : (N, ) list of float
        Edge width for each shape.
    z_indices : (N, ) list of int
        z-index for each shape.

    Notes
    -----
    _vertices : np.ndarray
        MxN array of all displayed vertices from all shapes where N is equal to ndisplay
    _index : np.ndarray
        Length M array with the index (0, ..., N-1) of each shape that each
        vertex corresponds to
    _z_index : np.ndarray
        Length N array with z_index of each shape
    _z_order : np.ndarray
        Length N array with z_order of each shape. This must be a permutation
        of (0, ..., N-1).
    _mesh : Mesh
        Mesh object containing all the mesh information that will ultimately
        be rendered.
    """

    def __init__(
        self, data: typing.Iterable[Shape] = (), ndisplay: int = 2
    ) -> None:
        self._ndisplay = ndisplay
        self.shapes: list[Shape] = []
        self._displayed = np.array([])
        self._slice_key = np.array([])
        self.displayed_vertices = np.array([])
        self.displayed_index = np.array([])
        self._vertices = np.empty((0, self.ndisplay))
        self._index = np.empty((0), dtype=int)
        self._z_index = np.empty((0), dtype=int)
        self._z_order = np.empty((0), dtype=int)

        self._mesh = Mesh(ndisplay=self.ndisplay)

        self._edge_color = np.empty((0, 4))
        self._face_color = np.empty((0, 4))

        # counter for the depth of re entrance of the context manager.
        self.__batched_level = 0
        self.__batch_force_call = False

        # Counter of number of time _update_displayed has been requested
        self.__update_displayed_called = 0

        for d in data:
            self.add(d)

    @contextmanager
    def batched_updates(self) -> Generator[None, None, None]:
        """
        Reentrant context manager to batch the display update

        `_update_displayed()` is called at _most_ once on exit of the context
        manager.

        There are two reason for this:

         1. Some updates are triggered by events, but sometimes multiple pieces
            of data that trigger events must be set before the data can be
            recomputed. For example changing number of dimension cause broacast
            error on partially update structures.
         2. Performance. Ideally we want to update the display only once.

        If no direct or indirect call to `_update_displayed()` are made inside
        the context manager, no the update logic is not called on exit.



        """
        assert self.__batched_level >= 0
        self.__batched_level += 1
        try:
            yield
        finally:
            if self.__batched_level == 1 and self.__update_displayed_called:
                self.__batch_force_call = True
                self._update_displayed()
                self.__batch_force_call = False
                self.__update_displayed_called = 0
            self.__batched_level -= 1

        assert self.__batched_level >= 0

    @property
    def data(self) -> list[npt.NDArray]:
        """list of (M, D) array: data arrays for each shape."""
        return [s.data for s in self.shapes]

    @property
    def ndisplay(self) -> int:
        """int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: int) -> None:
        if self.ndisplay == ndisplay:
            return

        self._ndisplay = ndisplay
        self._mesh.ndisplay = self.ndisplay
        self._vertices = np.empty((0, self.ndisplay))
        self._index = np.empty((0), dtype=int)
        for index in range(len(self.shapes)):
            shape = self.shapes[index]
            shape.ndisplay = self.ndisplay
            self.remove(index, renumber=False)
            self.add(shape, shape_index=index)
        self._update_z_order()

    @property
    def slice_keys(self) -> npt.NDArray:
        """(N, 2, P) array: slice key for each shape."""
        return np.array([s.slice_key for s in self.shapes])

    @property
    def shape_types(self) -> list[str]:
        """list of str: shape types for each shape."""
        return [s.name for s in self.shapes]

    @property
    def edge_color(self) -> npt.NDArray:
        """(N x 4) np.ndarray: Array of RGBA edge colors for each shape"""
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: npt.NDArray) -> None:
        self._set_color(edge_color, 'edge')

    @property
    def face_color(self) -> npt.NDArray:
        """(N x 4) np.ndarray: Array of RGBA face colors for each shape"""
        return self._face_color

    @face_color.setter
    def face_color(self, face_color: npt.NDArray) -> None:
        self._set_color(face_color, 'face')

    @_batch_dec
    def _set_color(
        self, colors: npt.NDArray, attribute: Literal['edge', 'face']
    ) -> None:
        """Set the face_color or edge_color property

        Parameters
        ----------
        colors : (N, 4) np.ndarray
            The value for setting edge or face_color. There must
            be one color for each shape
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        """
        n_shapes = len(self.data)
        if not np.array_equal(colors.shape, (n_shapes, 4)):
            raise ValueError(
                trans._(
                    '{attribute}_color must have shape ({n_shapes}, 4)',
                    deferred=True,
                    attribute=attribute,
                    n_shapes=n_shapes,
                )
            )

        update_method = getattr(self, f'update_{attribute}_colors')
        indices = np.arange(len(colors))
        update_method(indices, colors, update=False)
        self._update_displayed()

    @property
    def edge_widths(self) -> list[float]:
        """list of float: edge width for each shape."""
        return [s.edge_width for s in self.shapes]

    @property
    def z_indices(self) -> list[int]:
        """list of int: z-index for each shape."""
        return [s.z_index for s in self.shapes]

    @property
    def slice_key(self):
        """list: slice key for slicing n-dimensional shapes."""
        return self._slice_key

    @slice_key.setter
    @_batch_dec
    def slice_key(self, slice_key):
        slice_key = list(slice_key)
        if not np.array_equal(self._slice_key, slice_key):
            self._slice_key = slice_key
            self._clear_cache()
            self._update_displayed()

    def _update_displayed(self) -> None:
        """Update the displayed data based on the slice key.

        This method must be called from within the `batched_updates` context
        manager:
        """
        assert (
            self.__batched_level >= 1
        ), 'call _update_displayed from within self.batched_updates context manager'
        if not self.__batch_force_call:
            self.__update_displayed_called += 1
            return

        # The list slice key is repeated to check against both the min and
        # max values stored in the shapes slice key.
        slice_key = np.array([self.slice_key, self.slice_key])

        # Slice key must exactly match mins and maxs of shape as then the
        # shape is entirely contained within the current slice.
        if len(self.shapes) > 0:
            self._displayed = np.all(
                np.abs(self.slice_keys - slice_key) < 0.5, axis=(1, 2)
            )
        else:
            self._displayed = np.array([])
        disp_indices = np.where(self._displayed)[0]

        z_order = self._mesh.triangles_z_order
        disp_tri = np.isin(
            self._mesh.triangles_index[z_order, 0], disp_indices
        )
        self._mesh.displayed_triangles = self._mesh.triangles[z_order][
            disp_tri
        ]
        self._mesh.displayed_triangles_index = self._mesh.triangles_index[
            z_order
        ][disp_tri]
        self._mesh.displayed_triangles_colors = self._mesh.triangles_colors[
            z_order
        ][disp_tri]

        disp_vert = np.isin(self._index, disp_indices)
        self.displayed_vertices = self._vertices[disp_vert]
        self.displayed_index = self._index[disp_vert]

    def add(
        self,
        shape: Shape | Sequence[Shape],
        face_color=None,
        edge_color=None,
        shape_index=None,
        z_refresh=True,
    ):
        """Adds a single Shape object (single add mode) or multiple Shapes (multiple shape mode, which is much faster)

        If shape is a single instance of subclass Shape then single add mode will be used, otherwise multiple add mode

        Parameters
        ----------
        shape : single Shape or iterable of Shape
            Each shape must be a subclass of Shape, one of "{'Line', 'Rectangle',
            'Ellipse', 'Path', 'Polygon'}"
        face_color : color (or iterable of colors of same length as shape)
        edge_color : color (or iterable of colors of same length as shape)
        shape_index : None | int
            If int then edits the shape date at current index. To be used in
            conjunction with `remove` when renumber is `False`. If None, then
            appends a new shape to end of shapes list
            Must be None in multiple shape mode.
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When shape_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of shapes, set to false  and then call
            ShapesList._update_z_order() once at the end.
        """
        # single shape mode
        if issubclass(type(shape), Shape):
            self._add_single_shape(
                shape=shape,
                face_color=face_color,
                edge_color=edge_color,
                shape_index=shape_index,
                z_refresh=z_refresh,
            )
        # multiple shape mode
        elif isinstance(shape, Iterable):
            if shape_index is not None:
                raise ValueError(
                    trans._(
                        'shape_index must be None when adding multiple shapes',
                        deferred=True,
                    )
                )
            self._add_multiple_shapes(
                shapes=shape,
                face_colors=face_color,
                edge_colors=edge_color,
                z_refresh=z_refresh,
            )
        else:
            raise TypeError(
                trans._(
                    'Cannot add single nor multiple shape',
                    deferred=True,
                )
            )

    def _add_single_shape(
        self,
        shape,
        face_color=None,
        edge_color=None,
        shape_index=None,
        z_refresh=True,
    ):
        """Adds a single Shape object

        Parameters
        ----------
        shape : subclass Shape
            Must be a subclass of Shape, one of "{'Line', 'Rectangle',
            'Ellipse', 'Path', 'Polygon'}"
        shape_index : None | int
            If int then edits the shape date at current index. To be used in
            conjunction with `remove` when renumber is `False`. If None, then
            appends a new shape to end of shapes list
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When shape_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of shapes, set to false  and then call
            ShapesList._update_z_order() once at the end.
        """
        if not issubclass(type(shape), Shape):
            raise TypeError(
                trans._(
                    'shape must be subclass of Shape',
                    deferred=True,
                )
            )

        if shape_index is None:
            shape_index = len(self.shapes)
            self.shapes.append(shape)
            self._z_index = np.append(self._z_index, shape.z_index)

            if face_color is None:
                face_color = np.array([1, 1, 1, 1])
            self._face_color = np.vstack([self._face_color, face_color])
            if edge_color is None:
                edge_color = np.array([0, 0, 0, 1])
            self._edge_color = np.vstack([self._edge_color, edge_color])
        else:
            z_refresh = False
            self.shapes[shape_index] = shape
            self._z_index[shape_index] = shape.z_index

            if face_color is None:
                face_color = self._face_color[shape_index]
            else:
                self._face_color[shape_index, :] = face_color
            if edge_color is None:
                edge_color = self._edge_color[shape_index]
            else:
                self._edge_color[shape_index, :] = edge_color

        self._vertices = np.append(
            self._vertices, shape.data_displayed, axis=0
        )
        index = np.repeat(shape_index, len(shape.data))
        self._index = np.append(self._index, index, axis=0)

        # Add faces to mesh
        m = len(self._mesh.vertices)
        vertices = shape._face_vertices
        self._mesh.vertices = np.append(self._mesh.vertices, vertices, axis=0)
        vertices = shape._face_vertices
        self._mesh.vertices_centers = np.append(
            self._mesh.vertices_centers, vertices, axis=0
        )
        vertices = np.zeros(shape._face_vertices.shape)
        self._mesh.vertices_offsets = np.append(
            self._mesh.vertices_offsets, vertices, axis=0
        )
        index = np.repeat([[shape_index, 0]], len(vertices), axis=0)
        self._mesh.vertices_index = np.append(
            self._mesh.vertices_index, index, axis=0
        )

        triangles = shape._face_triangles + m
        self._mesh.triangles = np.append(
            self._mesh.triangles, triangles, axis=0
        )
        index = np.repeat([[shape_index, 0]], len(triangles), axis=0)
        self._mesh.triangles_index = np.append(
            self._mesh.triangles_index, index, axis=0
        )
        color_array = np.repeat([face_color], len(triangles), axis=0)
        self._mesh.triangles_colors = np.append(
            self._mesh.triangles_colors, color_array, axis=0
        )

        # Add edges to mesh
        m = len(self._mesh.vertices)
        vertices = (
            shape._edge_vertices + shape.edge_width * shape._edge_offsets
        )
        self._mesh.vertices = np.append(self._mesh.vertices, vertices, axis=0)
        vertices = shape._edge_vertices
        self._mesh.vertices_centers = np.append(
            self._mesh.vertices_centers, vertices, axis=0
        )
        vertices = shape._edge_offsets
        self._mesh.vertices_offsets = np.append(
            self._mesh.vertices_offsets, vertices, axis=0
        )
        index = np.repeat([[shape_index, 1]], len(vertices), axis=0)
        self._mesh.vertices_index = np.append(
            self._mesh.vertices_index, index, axis=0
        )

        triangles = shape._edge_triangles + m
        self._mesh.triangles = np.append(
            self._mesh.triangles, triangles, axis=0
        )
        index = np.repeat([[shape_index, 1]], len(triangles), axis=0)
        self._mesh.triangles_index = np.append(
            self._mesh.triangles_index, index, axis=0
        )
        color_array = np.repeat([edge_color], len(triangles), axis=0)
        self._mesh.triangles_colors = np.append(
            self._mesh.triangles_colors, color_array, axis=0
        )

        if z_refresh:
            # Set z_order
            self._update_z_order()
        self._clear_cache()

    def _add_multiple_shapes(
        self,
        shapes,
        face_colors=None,
        edge_colors=None,
        z_refresh=True,
    ):
        """Add multiple shapes at once (faster than adding them one by one)

        Parameters
        ----------
        shapes : iterable of Shape
            Each Shape must be a subclass of Shape, one of "{'Line', 'Rectangle',
            'Ellipse', 'Path', 'Polygon'}"
        face_colors : iterable of face_color
        edge_colors : iterable of edge_color
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When shape_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of shapes, set to false  and then call
            ShapesList._update_z_order() once at the end.

        TODO: Currently shares a lot of code with `add()`, with the
        difference being that `add()` supports inserting shapes at a specific
        `shape_index`, whereas `add_multiple` will append them as a full batch
        """

        def _make_index(length, shape_index, cval=0):
            """Same but faster than `np.repeat([[shape_index, cval]], length, axis=0)`"""
            index = np.empty((length, 2), np.int32)
            index.fill(cval)
            index[:, 0] = shape_index
            return index

        all_z_index = []
        all_vertices = []
        all_index = []
        all_mesh_vertices = []
        all_mesh_vertices_centers = []
        all_mesh_vertices_offsets = []
        all_mesh_vertices_index = []
        all_mesh_triangles = []
        all_mesh_triangles_index = []
        all_mesh_triangles_colors = []

        m_mesh_vertices_count = len(self._mesh.vertices)

        if face_colors is None:
            face_colors = np.tile(np.array([1, 1, 1, 1]), (len(shapes), 1))
        else:
            face_colors = np.asarray(face_colors)

        if edge_colors is None:
            edge_colors = np.tile(np.array([0, 0, 0, 1]), (len(shapes), 1))
        else:
            edge_colors = np.asarray(edge_colors)

        if not len(face_colors) == len(edge_colors) == len(shapes):
            raise ValueError(
                trans._(
                    'shapes, face_colors, and edge_colors must be the same length',
                    deferred=True,
                )
            )

        if not all(issubclass(type(shape), Shape) for shape in shapes):
            raise ValueError(
                trans._(
                    'all shapes must be subclass of Shape',
                    deferred=True,
                )
            )

        for shape, face_color, edge_color in zip(
            shapes, face_colors, edge_colors, strict=False
        ):
            shape_index = len(self.shapes)
            self.shapes.append(shape)
            all_z_index.append(shape.z_index)
            all_vertices.append(shape.data_displayed)
            all_index.append([shape_index] * len(shape.data))

            # Add faces to mesh
            m_tmp = m_mesh_vertices_count
            all_mesh_vertices.append(shape._face_vertices)
            m_mesh_vertices_count += len(shape._face_vertices)
            all_mesh_vertices_centers.append(shape._face_vertices)
            vertices = np.zeros(shape._face_vertices.shape)
            all_mesh_vertices_offsets.append(vertices)
            all_mesh_vertices_index.append(
                _make_index(len(vertices), shape_index, cval=0)
            )

            triangles = shape._face_triangles + m_tmp
            all_mesh_triangles.append(triangles)
            all_mesh_triangles_index.append(
                _make_index(len(triangles), shape_index, cval=0)
            )

            color_array = np.repeat([face_color], len(triangles), axis=0)
            all_mesh_triangles_colors.append(color_array)

            # Add edges to mesh
            m_tmp = m_mesh_vertices_count

            vertices = (
                shape._edge_vertices + shape.edge_width * shape._edge_offsets
            )
            all_mesh_vertices.append(vertices)
            m_mesh_vertices_count += len(vertices)

            all_mesh_vertices_centers.append(shape._edge_vertices)

            all_mesh_vertices_offsets.append(shape._edge_offsets)

            all_mesh_vertices_index.append(
                _make_index(len(shape._edge_offsets), shape_index, cval=1)
            )

            triangles = shape._edge_triangles + m_tmp
            all_mesh_triangles.append(triangles)
            all_mesh_triangles_index.append(
                _make_index(len(triangles), shape_index, cval=1)
            )

            color_array = np.repeat([edge_color], len(triangles), axis=0)
            all_mesh_triangles_colors.append(color_array)

        # assemble properties
        self._z_index = np.append(self._z_index, np.array(all_z_index), axis=0)
        self._face_color = np.vstack((self._face_color, face_colors))
        self._edge_color = np.vstack((self._edge_color, edge_colors))
        self._vertices = np.vstack((self._vertices, np.vstack(all_vertices)))
        self._index = np.append(self._index, np.concatenate(all_index), axis=0)

        self._mesh.vertices = np.vstack(
            (self._mesh.vertices, np.vstack(all_mesh_vertices))
        )
        self._mesh.vertices_centers = np.vstack(
            (self._mesh.vertices_centers, np.vstack(all_mesh_vertices_centers))
        )
        self._mesh.vertices_offsets = np.vstack(
            (self._mesh.vertices_offsets, np.vstack(all_mesh_vertices_offsets))
        )
        self._mesh.vertices_index = np.vstack(
            (self._mesh.vertices_index, np.vstack(all_mesh_vertices_index))
        )

        self._mesh.triangles = np.vstack(
            (self._mesh.triangles, np.vstack(all_mesh_triangles))
        )
        self._mesh.triangles_index = np.vstack(
            (self._mesh.triangles_index, np.vstack(all_mesh_triangles_index))
        )
        self._mesh.triangles_colors = np.vstack(
            (self._mesh.triangles_colors, np.vstack(all_mesh_triangles_colors))
        )

        if z_refresh:
            # Set z_order
            self._update_z_order()
        self._clear_cache()

    @_batch_dec
    def remove_all(self):
        """Removes all shapes"""
        self.shapes = []
        self._vertices = np.empty((0, self.ndisplay))
        self._index = np.empty((0), dtype=int)
        self._z_index = np.empty((0), dtype=int)
        self._z_order = np.empty((0), dtype=int)
        self._mesh.clear()
        self._update_displayed()

    def remove(self, index, renumber=True):
        """Removes a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be removed.
        renumber : bool
            Bool to indicate whether to renumber all shapes or not. If not the
            expectation is that this shape is being immediately added back to the
            list using `add_shape`.
        """
        indices = self._index != index
        self._vertices = self._vertices[indices]
        self._index = self._index[indices]

        # Remove triangles
        indices = self._mesh.triangles_index[:, 0] != index
        self._mesh.triangles = self._mesh.triangles[indices]
        self._mesh.triangles_colors = self._mesh.triangles_colors[indices]
        self._mesh.triangles_index = self._mesh.triangles_index[indices]

        # Remove vertices
        indices = self._mesh.vertices_index[:, 0] != index
        self._mesh.vertices = self._mesh.vertices[indices]
        self._mesh.vertices_centers = self._mesh.vertices_centers[indices]
        self._mesh.vertices_offsets = self._mesh.vertices_offsets[indices]
        self._mesh.vertices_index = self._mesh.vertices_index[indices]
        indices = np.where(np.invert(indices))[0]
        num_indices = len(indices)
        if num_indices > 0:
            indices = self._mesh.triangles > indices[0]
            self._mesh.triangles[indices] = (
                self._mesh.triangles[indices] - num_indices
            )

        if renumber:
            del self.shapes[index]
            indices = self._index > index
            self._index[indices] = self._index[indices] - 1
            self._z_index = np.delete(self._z_index, index)
            indices = self._mesh.triangles_index[:, 0] > index
            self._mesh.triangles_index[indices, 0] = (
                self._mesh.triangles_index[indices, 0] - 1
            )
            indices = self._mesh.vertices_index[:, 0] > index
            self._mesh.vertices_index[indices, 0] = (
                self._mesh.vertices_index[indices, 0] - 1
            )
            self._update_z_order()
        self._clear_cache()

    @_batch_dec
    def _update_mesh_vertices(self, index, edge=False, face=False):
        """Updates the mesh vertex data and vertex data for a single shape
        located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge : bool
            Bool to indicate whether to update mesh vertices corresponding to
            edges
        face : bool
            Bool to indicate whether to update mesh vertices corresponding to
            faces and to update the underlying shape vertices
        """
        shape = self.shapes[index]
        if edge:
            indices = np.all(self._mesh.vertices_index == [index, 1], axis=1)
            self._mesh.vertices[indices] = (
                shape._edge_vertices + shape.edge_width * shape._edge_offsets
            )
            self._mesh.vertices_centers[indices] = shape._edge_vertices
            self._mesh.vertices_offsets[indices] = shape._edge_offsets
            self._update_displayed()

        if face:
            indices = np.all(self._mesh.vertices_index == [index, 0], axis=1)
            self._mesh.vertices[indices] = shape._face_vertices
            self._mesh.vertices_centers[indices] = shape._face_vertices
            indices = self._index == index
            self._vertices[indices] = shape.data_displayed
            self._update_displayed()
        self._clear_cache()

    @_batch_dec
    def _update_z_order(self):
        """Updates the z order of the triangles given the z_index list"""
        self._z_order = np.argsort(self._z_index)
        if len(self._z_order) == 0:
            self._mesh.triangles_z_order = np.empty((0), dtype=int)
        else:
            _, idx, counts = np.unique(
                self._mesh.triangles_index[:, 0],
                return_index=True,
                return_counts=True,
            )
            triangles_z_order = [
                np.arange(idx[z], idx[z] + counts[z]) for z in self._z_order
            ]
            self._mesh.triangles_z_order = np.concatenate(triangles_z_order)
        self._update_displayed()

    def edit(
        self, index, data, face_color=None, edge_color=None, new_type=None
    ):
        """Updates the data of a single shape located at index. If
        `new_type` is not None then converts the shape type to the new type

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        data : np.ndarray
            NxD array of vertices.
        new_type : None | str | Shape
            If string , must be one of "{'line', 'rectangle', 'ellipse',
            'path', 'polygon'}".
        """
        if new_type is not None:
            cur_shape = self.shapes[index]
            if isinstance(new_type, str):
                shape_type = ShapeType(new_type)
                if shape_type in shape_classes:
                    shape_cls = shape_classes[shape_type]
                else:
                    raise ValueError(
                        trans._(
                            '{shape_type} must be one of {shape_classes}',
                            deferred=True,
                            shape_type=shape_type,
                            shape_classes=set(shape_classes),
                        )
                    )
            else:
                shape_cls = new_type
            shape = shape_cls(
                data,
                edge_width=cur_shape.edge_width,
                z_index=cur_shape.z_index,
                dims_order=cur_shape.dims_order,
            )
        else:
            shape = self.shapes[index]
            shape.data = data

        if face_color is not None:
            self._face_color[index] = face_color
        if edge_color is not None:
            self._edge_color[index] = edge_color

        self.remove(index, renumber=False)
        self.add(shape, shape_index=index)
        self._update_z_order()

    def update_edge_width(self, index, edge_width):
        """Updates the edge width of a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge_width : float
            thickness of lines and edges.
        """
        self.shapes[index].edge_width = edge_width
        self._update_mesh_vertices(index, edge=True)

    @_batch_dec
    def update_edge_color(self, index, edge_color, update=True):
        """Updates the edge color of a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        edge_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
        update : bool
            If True, update the mesh with the new color property. Set to False to avoid
            repeated updates when modifying multiple shapes. Default is True.
        """
        self._edge_color[index] = edge_color
        indices = np.all(self._mesh.triangles_index == [index, 1], axis=1)
        self._mesh.triangles_colors[indices] = self._edge_color[index]
        if update:
            self._update_displayed()

    @_batch_dec
    def update_edge_colors(self, indices, edge_colors, update=True):
        """same as update_edge_color() but for multiple indices/edgecolors at once"""
        self._edge_color[indices] = edge_colors
        all_indices = np.bitwise_and(
            np.isin(self._mesh.triangles_index[:, 0], indices),
            self._mesh.triangles_index[:, 1] == 1,
        )
        self._mesh.triangles_colors[all_indices] = self._edge_color[
            self._mesh.triangles_index[all_indices, 0]
        ]
        if update:
            self._update_displayed()

    @_batch_dec
    def update_face_color(self, index, face_color, update=True):
        """Updates the face color of a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        face_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
        update : bool
            If True, update the mesh with the new color property. Set to False to avoid
            repeated updates when modifying multiple shapes. Default is True.
        """
        self._face_color[index] = face_color
        indices = np.all(self._mesh.triangles_index == [index, 0], axis=1)
        self._mesh.triangles_colors[indices] = self._face_color[index]
        if update:
            self._update_displayed()

    @_batch_dec
    def update_face_colors(self, indices, face_colors, update=True):
        """same as update_face_color() but for multiple indices/facecolors at once"""
        self._face_color[indices] = face_colors
        all_indices = np.bitwise_and(
            np.isin(self._mesh.triangles_index[:, 0], indices),
            self._mesh.triangles_index[:, 1] == 0,
        )
        self._mesh.triangles_colors[all_indices] = self._face_color[
            self._mesh.triangles_index[all_indices, 0]
        ]
        if update:
            self._update_displayed()

    def update_dims_order(self, dims_order):
        """Updates dimensions order for all shapes.

        Parameters
        ----------
        dims_order : (D,) list
            Order that the dimensions are rendered in.
        """
        for index in range(len(self.shapes)):
            if self.shapes[index].dims_order != dims_order:
                shape = self.shapes[index]
                shape.dims_order = dims_order
                self.remove(index, renumber=False)
                self.add(shape, shape_index=index)
        self._update_z_order()

    def update_z_index(self, index, z_index):
        """Updates the z order of a single shape located at index.

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        z_index : int
            Specifier of z order priority. Shapes with higher z order are
            displayed ontop of others.
        """
        self.shapes[index].z_index = z_index
        self._z_index[index] = z_index
        self._update_z_order()

    def shift(self, index, shift):
        """Performs a 2D shift on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        shift : np.ndarray
            length 2 array specifying shift of shapes.
        """
        self.shapes[index].shift(shift)
        self._update_mesh_vertices(index, edge=True, face=True)

    def scale(self, index, scale, center=None):
        """Performs a scaling on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : list
            length 2 list specifying coordinate of center of scaling.
        """
        self.shapes[index].scale(scale, center=center)
        shape = self.shapes[index]
        self.remove(index, renumber=False)
        self.add(shape, shape_index=index)
        self._update_z_order()

    def rotate(self, index, angle, center=None):
        """Performs a rotation on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        angle : float
            angle specifying rotation of shape in degrees.
        center : list
            length 2 list specifying coordinate of center of rotation.
        """
        self.shapes[index].rotate(angle, center=center)
        self._update_mesh_vertices(index, edge=True, face=True)

    def flip(self, index, axis, center=None):
        """Performs an vertical flip on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        axis : int
            integer specifying axis of flip. `0` flips horizontal, `1` flips
            vertical.
        center : list
            length 2 list specifying coordinate of center of flip axes.
        """
        self.shapes[index].flip(axis, center=center)
        self._update_mesh_vertices(index, edge=True, face=True)

    def transform(self, index, transform):
        """Performs a linear transform on a single shape located at index

        Parameters
        ----------
        index : int
            Location in list of the shape to be changed.
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        self.shapes[index].transform(transform)
        shape = self.shapes[index]
        self.remove(index, renumber=False)
        self.add(shape, shape_index=index)
        self._update_z_order()
        self._clear_cache()

    def outline(
        self, indices: int | Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finds outlines of shapes listed in indices

        Parameters
        ----------
        indices : int | Sequence[int]
            Location in list of the shapes to be outline.
            If sequence, all elements should be ints

        Returns
        -------
        centers : np.ndarray
            Nx2 array of centers of outline
        offsets : np.ndarray
            Nx2 array of offsets of outline
        triangles : np.ndarray
            Mx3 array of any indices of vertices for triangles of outline
        """
        if isinstance(indices, Sequence) and len(indices) == 1:
            indices = indices[0]
        if not isinstance(indices, Sequence):
            shape = self.shapes[indices]
            return (
                shape._edge_vertices,
                shape._edge_offsets,
                shape._edge_triangles,
            )
        return self.outlines(indices)

    def outlines(
        self, indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finds outlines of shapes listed in indices

        Parameters
        ----------
        indices : Sequence[int]
            Location in list of the shapes to be outline.

        Returns
        -------
        centers : np.ndarray
            Nx2 array of centers of outline
        offsets : np.ndarray
            Nx2 array of offsets of outline
        triangles : np.ndarray
            Mx3 array of any indices of vertices for triangles of outline
        """
        shapes_list = [self.shapes[i] for i in indices]
        offsets = np.vstack([s._edge_offsets for s in shapes_list])
        centers = np.vstack([s._edge_vertices for s in shapes_list])
        vert_count = np.cumsum(
            [0] + [len(s._edge_vertices) for s in shapes_list]
        )
        triangles = np.vstack(
            [
                s._edge_triangles + c
                for s, c in zip(shapes_list, vert_count, strict=False)
            ]
        )

        return centers, offsets, triangles

    def shapes_in_box(self, corners):
        """Determines which shapes, if any, are inside an axis aligned box.

        Looks only at displayed shapes

        Parameters
        ----------
        corners : np.ndarray
            2x2 array of two corners that will be used to create an axis
            aligned box.

        Returns
        -------
        shapes : list
            List of shapes that are inside the box.
        """

        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        intersects = triangles_intersect_box(triangles, corners)
        shapes = self._mesh.displayed_triangles_index[intersects, 0]
        shapes = np.unique(shapes).tolist()

        return shapes

    @cached_property
    def _visible_shapes(self):
        slice_key = self.slice_key
        if len(slice_key):
            return [
                (i, s)
                for i, s in enumerate(self.shapes)
                if s.slice_key[0] <= slice_key <= s.slice_key[1]
            ]
        return list(enumerate(self.shapes))

    @cached_property
    def _bounding_boxes(self):
        data = np.array([s[1].bounding_box for s in self._visible_shapes])
        if data.size == 0:
            return np.empty((0, self.ndisplay)), np.empty((0, self.ndisplay))
        return data[:, 0], data[:, 1]

    def inside(self, coord):
        """Determines if any shape at given coord by looking inside triangle
        meshes. Looks only at displayed shapes

        Parameters
        ----------
        coord : sequence of float
            Image coordinates to check if any shapes are at.

        Returns
        -------
        shape : int | None
            Index of shape if any that is at the coordinates. Returns `None`
            if no shape is found.
        """
        if not self.shapes:
            return None
        bounding_boxes = self._bounding_boxes
        in_bbox = np.all(
            (bounding_boxes[0] <= coord) * (bounding_boxes[1] >= coord),
            axis=1,
        )
        inside_indices = np.flatnonzero(in_bbox)
        if inside_indices.size == 0:
            return None
        try:
            z_index = [
                self._visible_shapes[i][1].z_index for i in inside_indices
            ]
            pos = np.argsort(z_index)
            return self._visible_shapes[
                next(
                    inside_indices[p]
                    for p in pos[::-1]
                    if np.any(
                        inside_triangles(
                            self._visible_shapes[inside_indices[p]][
                                1
                            ]._all_triangles()
                            - coord
                        )
                    )
                )
            ][0]
        except StopIteration:
            return None

    def _inside_3d(self, ray_position: np.ndarray, ray_direction: np.ndarray):
        """Determines if any shape is intersected by a ray by looking inside triangle
        meshes. Looks only at displayed shapes.

        Parameters
        ----------
        ray_position : np.ndarray
            (3,) array containing the location that was clicked. This
            should be in the same coordinate system as the vertices.
        ray_direction : np.ndarray
            (3,) array describing the direction camera is pointing in
            the scene. This should be in the same coordinate system as
            the vertices.

        Returns
        -------
        shape : int | None
            Index of shape if any that is at the coordinates. Returns `None`
            if no shape is found.
        intersection_point : Optional[np.ndarray]
            The point where the ray intersects the mesh face. If there was
            no intersection, returns None.
        """
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        inside = line_in_triangles_3d(
            line_point=ray_position,
            line_direction=ray_direction,
            triangles=triangles,
        )
        intersected_shapes = self._mesh.displayed_triangles_index[inside, 0]
        if len(intersected_shapes) == 0:
            return None, None

        intersection_points = self._triangle_intersection(
            triangle_indices=inside,
            ray_position=ray_position,
            ray_direction=ray_direction,
        )
        start_to_intersection = intersection_points - ray_position
        distances = np.linalg.norm(start_to_intersection, axis=1)
        closest_shape_index = np.argmin(distances)
        shape = intersected_shapes[closest_shape_index]
        intersection = intersection_points[closest_shape_index]
        return shape, intersection

    def _triangle_intersection(
        self,
        triangle_indices: np.ndarray,
        ray_position: np.ndarray,
        ray_direction: np.ndarray,
    ):
        """Find the intersection of a ray with specified triangles.

        Parameters
        ----------
        triangle_indices : np.ndarray
            (n,) array of shape indices to find the intersection with the ray. The indices should
            correspond with self._mesh.displayed_triangles.
        ray_position : np.ndarray
            (3,) array with the coordinate of the starting point of the ray in layer coordinates.
            Only provide the 3 displayed dimensions.
        ray_direction : np.ndarray
            (3,) array of the normal direction of the ray in layer coordinates.
            Only provide the 3 displayed dimensions.

        Returns
        -------
        intersection_points : np.ndarray
            (n x 3) array of the intersection of the ray with each of the specified shapes in layer coordinates.
            Only the 3 displayed dimensions are provided.
        """
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        intersected_triangles = triangles[triangle_indices]
        intersection_points = intersect_line_with_triangles(
            line_point=ray_position,
            line_direction=ray_direction,
            triangles=intersected_triangles,
        )
        return intersection_points

    def to_masks(self, mask_shape=None, zoom_factor=1, offset=(0, 0)):
        """Returns N binary masks, one for each shape, embedded in an array of
        shape `mask_shape`.

        Parameters
        ----------
        mask_shape : np.ndarray | tuple | None
            2-tuple defining shape of mask to be generated. If non specified,
            takes the max of all the vertices
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.

        Returns
        -------
        masks : (N, M, P) np.ndarray
            Array where there is one binary mask of shape MxP for each of
            N shapes
        """
        if mask_shape is None:
            mask_shape = self.displayed_vertices.max(axis=0).astype('int')

        masks = np.array(
            [
                s.to_mask(mask_shape, zoom_factor=zoom_factor, offset=offset)
                for s in self.shapes
            ]
        )

        return masks

    def to_labels(self, labels_shape=None, zoom_factor=1, offset=(0, 0)):
        """Returns a integer labels image, where each shape is embedded in an
        array of shape labels_shape with the value of the index + 1
        corresponding to it, and 0 for background. For overlapping shapes
        z-ordering will be respected.

        Parameters
        ----------
        labels_shape : np.ndarray | tuple | None
            2-tuple defining shape of labels image to be generated. If non
            specified, takes the max of all the vertices
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.

        Returns
        -------
        labels : np.ndarray
            MxP integer array where each value is either 0 for background or an
            integer up to N for points inside the corresponding shape.
        """
        if labels_shape is None:
            labels_shape = self.displayed_vertices.max(axis=0).astype(int)

        labels = np.zeros(labels_shape, dtype=int)

        for ind in self._z_order[::-1]:
            mask = self.shapes[ind].to_mask(
                labels_shape, zoom_factor=zoom_factor, offset=offset
            )
            labels[mask] = ind + 1

        return labels

    def to_colors(
        self, colors_shape=None, zoom_factor=1, offset=(0, 0), max_shapes=None
    ):
        """Rasterize shapes to an RGBA image array.

        Each shape is embedded in an array of shape `colors_shape` with the
        RGBA value of the shape, and 0 for background. For overlapping shapes
        z-ordering will be respected.

        Parameters
        ----------
        colors_shape : np.ndarray | tuple | None
            2-tuple defining shape of colors image to be generated. If non
            specified, takes the max of all the vertiecs
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.
        max_shapes : None | int
            If provided, this is the maximum number of shapes that will be rasterized.
            If the number of shapes in view exceeds max_shapes, max_shapes shapes
            will be randomly selected from the in view shapes. If set to None, no
            maximum is applied. The default value is None.

        Returns
        -------
        colors : (N, M, 4) array
            rgba array where each value is either 0 for background or the rgba
            value of the shape for points inside the corresponding shape.
        """
        if colors_shape is None:
            colors_shape = self.displayed_vertices.max(axis=0).astype(int)

        colors = np.zeros((*colors_shape, 4), dtype=float)
        colors[..., 3] = 1

        z_order = self._z_order[::-1]
        shapes_in_view = np.argwhere(self._displayed)
        z_order_in_view_mask = np.isin(z_order, shapes_in_view)
        z_order_in_view = z_order[z_order_in_view_mask]

        # If there are too many shapes to render responsively, just render
        # the top max_shapes shapes
        if max_shapes is not None and len(z_order_in_view) > max_shapes:
            z_order_in_view = z_order_in_view[:max_shapes]

        for ind in z_order_in_view:
            mask = self.shapes[ind].to_mask(
                colors_shape, zoom_factor=zoom_factor, offset=offset
            )
            if type(self.shapes[ind]) in [Path, Line]:
                col = self._edge_color[ind]
            else:
                col = self._face_color[ind]
            colors[mask, :] = col

        return colors

    def _clear_cache(self):
        self.__dict__.pop('_bounding_boxes', None)
        self.__dict__.pop('_visible_shapes', None)
