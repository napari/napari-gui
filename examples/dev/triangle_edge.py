"""Visualize the triangulation algorithms used by the Shapes layer.

This example uses napari layers to draw each of the components of a face and
edge triangulation in a Shapes layer.

Shapes layers don't "just" draw polygons, ellipses, and so on: each shape,
as well as its borders, is broken down into a collection of triangles (this
is called a *triangulation*), which are then sent to OpenGL for drawing:
drawing triangles is one of the "visualization primitives" in OpenGL and most
2D and 3D drawing frameworks.

It turns out that converting arbitrary shapes into a collection of triangles
can be quite tricky: very shallow angles cause errors in the algorithms, and
can also make certain desirable properties (such as edges not overlapping with
each other when a polygon makes a sharp turn) actually impossible to achieve
with fast (single-pass) algorithms.

This example draws the Shapes layer normally, but also overlays all the
elements of the triangulation: the triangles themselves, and the normal vectors
on each polygon vertex, from which the triangulation is computed.
"""

from dataclasses import dataclass, fields
from functools import partial

try:
    import numba
except ImportError:
    # Create a dummy numba.njit to allow running this script without numba
    import toolz as tz
    class numba:
        @tz.curry
        def njit(func, cache=True, inline=None):
            return func

import numpy as np

import napari
from napari.layers import Shapes
from napari.layers.shapes._shapes_utils import generate_2D_edge_meshes


def generate_regular_polygon(n, radius=1):
    """Generate vertices of a regular n-sided polygon centered at the origin.

    Parameters
    ----------
    n : int
        The number of sides (vertices).
    radius : float, optional
        The radius of the circumscribed circle.

    Returns
    -------
    np.ndarray
        An array of shape (n, 2) containing the vertex coordinates.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def get_reference_edge_triangulation_points(shape: Shapes) -> np.ndarray:
    """Get the non-accelerated points"""
    shapes = shape._data_view.shapes
    path_list = [(x.data, x._closed) for x in shapes]
    mesh_list = [generate_2D_edge_meshes(path, closed)
                 for path, closed in path_list]
    mesh = tuple(np.concatenate(el, axis=0) for el in zip(*mesh_list, strict=False))

    return mesh[0] + mesh[1]


def generate_order_vectors(path_, closed) -> np.ndarray:
    """Generate the vectors tangent to the path.

    Parameters
    ----------
    path_ : np.ndarray, shape (N, 2)
        A list of 2D path coordinates.
    closed : bool
        Whether the coordinates represent a closed polygon or an open
        line/path.

    Returns
    -------
    vec : np.ndarray, shape (N, 2, 2)
        A set of vectors, defined by a 2D position and a 2D projection.
    """
    raw_vecs = np.diff(path_, axis=0)
    norm = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    directions = raw_vecs / norm
    vec = np.empty((path_.shape[0], 2, 2))
    vec[:, 0] = path_
    vec[:-1, 1] = directions
    if closed:
        # point from the last vertex towards the first vertex
        vec[-1, 1] = (
                (path_[0] - path_[-1]) / np.linalg.norm(path_[-1] - path_[0])
                )
    else:
        # point back towards the penultimate vertex
        vec[-1, 1] = -vec[-2, 1]
    return vec


def generate_miter_helper_vectors(
        direction_vectors_array: np.ndarray
        ) -> np.ndarray:
    """Generate the miter helper vectors.

    The miter helper vectors are a pair of vectors for each point in the path,
    which together help define whether a bevel join will be needed. The first
    vector is half of the normalized direction vector for that vertex, and the
    second is *minus* half of the normalized direction vector for the previous
    vertex. Their angle is the angle of the edges at that vertex.

    Parameters
    ----------
    direction_vectors_array : array of shape (n, 2)
        array of normalized (length 1) direction vectors for each point in the
        path.

    Returns
    -------
    array of shape (n, 2, 2)
        array of miter helper vectors
    """
    half_direction = direction_vectors_array.copy()
    half_direction[:, 1] *= 0.5
    half_prev_direction = half_direction.copy()
    half_prev_direction[:, 1] *= -1
    half_prev_direction[:, 1] = np.roll(half_prev_direction[:, 1], 1, axis=0)
    half_prev_direction[:, 0] += half_direction[:, 1]
    return np.concatenate([half_direction, half_prev_direction], axis=0)


@numba.njit
def generate_orthogonal_vectors(direction_vectors: np.ndarray) -> np.ndarray:
    """Generate the orthogonal vectors to the direction vectors.

    The orthogonal vector starts at the middle of the direction vector and is
    orthogonal to it in the positive orientation. Its length is half of the
    direction vector, to align with the normalized edge width.

    Parameters
    ----------
    direction_vectors : array, shape (n, 2, 2)
        The direction vector start points (``direction_vectors[:, 0, :]``) and
        directions (``direction_vectors[:, 1, :]``).

    Returns
    -------
    orthogonal_vectors : array, shape(n, 2, 2)
        The orthogonal vector start points and directions.
    """
    position = 0
    vector = 1
    y, x = 0, 1
    half_direction = 0.5 * direction_vectors[:, 1, :]
    orthogonal_vectors = direction_vectors.copy()
    orthogonal_vectors[:, position] = (
            direction_vectors[:, position] + half_direction
            )

    orthogonal_vectors[:, vector, y] = -half_direction[:, x]
    orthogonal_vectors[:, vector, x] = half_direction[:, y]
    return orthogonal_vectors


@numba.njit
def generate_miter_vectors(
        mesh: tuple[np.ndarray, np.ndarray, np.ndarray]
        ) -> np.ndarray:
    """For each point on path, generate the vectors pointing to miter points.

    Parameters
    ----------
    mesh : tuple[np.ndarray]
        vertices, offsets, and triangles of the mesh corresponding to the edges
        of a single shape.

    Returns
    -------
    np.ndarray, shape (n, 2, 2)
        Positions and projections of vectors corresponding to the miter points
        offset from the path points.
    """
    vec_points = np.empty((mesh[0].shape[0], 2, 2))
    vec_points[:, 0] = mesh[0]
    vec_points[:, 1] = mesh[1]
    return vec_points


@numba.njit
def generate_edge_triangle_borders(centers, offsets, triangles) -> np.ndarray:
    """Generate 3 vectors that represent the borders of the triangle.

    These are vectors to show the *ordering* of the triangles in the data.

    Parameters
    ----------
    centers, offsets, triangles : np.ndarray of float
        The mesh triangulation of the shape's edge path.

    Returns
    -------
    borders : np.ndarray of float
        Positions and projections corresponding to each triangle edge in
        the triangulation of the path.
    """
    borders = np.empty((len(triangles)*3, 2, 2), dtype=centers.dtype)
    position = 0
    projection = 1
    for i, triangle in enumerate(triangles):
        a, b, c = triangle
        a1 = centers[a] + offsets[a]
        b1 = centers[b] + offsets[b]
        c1 = centers[c] + offsets[c]
        borders[i * 3, position] = a1
        borders[i * 3, projection] = (b1 - a1)
        borders[i * 3 + 1, position] = b1
        borders[i * 3 + 1, projection] = (c1 - b1)
        borders[i * 3 + 2, position] = c1
        borders[i * 3 + 2, projection] = (a1 - c1)
    return borders


@numba.njit
def generate_face_triangle_borders(vertices, triangles) -> np.ndarray:
    """For each triangle in mesh generate 3 vectors that represent the borders of the triangle.
    """
    borders = np.empty((len(triangles)*3, 2, 2), dtype=vertices.dtype)
    for i, triangle in enumerate(triangles):
        a, b, c = triangle
        a1 = vertices[a]
        b1 = vertices[b]
        c1 = vertices[c]
        borders[i * 3, 0] = a1
        borders[i * 3, 1] = (b1 - a1)
        borders[i * 3 + 1, 0] = b1
        borders[i * 3 + 1, 1] = (c1 - b1)
        borders[i * 3 + 2, 0] = c1
        borders[i * 3 + 2, 1] = (a1 - c1)
    return borders


@dataclass
class Helpers:
    """Simple class to hold all auxiliary vector data for a shapes layer."""
    reference_join_points: np.ndarray
    join_points: np.ndarray
    direction_vectors: np.ndarray
    miter_helper_vectors: np.ndarray
    orthogonal_vectors: np.ndarray
    miter_vectors: np.ndarray
    triangles_vectors: np.ndarray
    face_triangles_vectors: np.ndarray


def asdict(dataclass_instance):
    """Shallow copy version of `dataclasses.asdict`."""
    return {f.name: getattr(dataclass_instance, f.name)
            for f in fields(dataclass_instance)}


def get_helper_data_from_shapes(shapes_layer: Shapes) -> Helpers:
    """Function to generate all auxiliary data for a shapes layer."""
    shapes = shapes_layer._data_view.shapes
    mesh_list = [(s._edge_vertices, s._edge_offsets, s._edge_triangles)
                 for s in shapes]
    path_list = [(s.data, s._closed) for s in shapes]
    mesh = tuple(np.concatenate(el, axis=0) for el in zip(*mesh_list, strict=False))
    face_mesh_list = [(s._face_vertices, s._face_triangles)
                      for s in shapes
                      if len(s._face_vertices) > 0]
    order_vectors_list = [generate_order_vectors(path_, closed)
                          for path_, closed in path_list]

    helpers = Helpers(
        reference_join_points=get_reference_edge_triangulation_points(
                shapes_layer
                ),
        join_points=mesh[0] + mesh[1],
        direction_vectors=np.concatenate(order_vectors_list, axis=0),
        miter_helper_vectors=np.concatenate(
                [generate_miter_helper_vectors(o) for o in order_vectors_list],
                axis=0,
                ),
        orthogonal_vectors=np.concatenate(
                [generate_orthogonal_vectors(o) for o in order_vectors_list],
                axis=0,
                ),
        miter_vectors=np.concatenate(
                [generate_miter_vectors(m) for m in mesh_list],
                axis=0,
                ),
        triangles_vectors=np.concatenate(
                [generate_edge_triangle_borders(*m) for m in mesh_list],
                axis=0,
                ),
        face_triangles_vectors=np.concatenate(
                [generate_face_triangle_borders(*m) for m in face_mesh_list],
                axis=0,
                ),
    )

    return helpers


def update_helper_layers(viewer: napari.Viewer, source_layer: Shapes):
    updated_helpers = get_helper_data_from_shapes(source_layer)
    for name, data in asdict(updated_helpers).items():
        viewer.layers[name].data = data


def add_helper_layers(viewer: napari.Viewer, source_layer):
    """Add helper layers to the viewer that track with the source shapes."""
    helpers = get_helper_data_from_shapes(source_layer)
    # sizes and colors are hardcoded based on vibes
    sizes = [0.2, 0.1, 0.1, 0.06, 0.04, 0.05, 0.04, 0.04]
    colors = ['yellow', 'white', 'red', 'blue',
              'green', 'yellow', 'pink', 'magenta']
    for (name, data), size, color in zip(
            asdict(helpers).items(), sizes, colors, strict=False
            ):
        if 'points' in name:
            viewer.add_points(data, name=name, size=size, face_color=color)
        else:  # all other fields are vectors
            viewer.add_vectors(
                    data,
                    name=name,
                    vector_style='arrow', edge_width=size, edge_color=color,
                    )
    source_layer.events.set_data.connect(
        partial(update_helper_layers, viewer=viewer, source_layer=source_layer)
    )


path = np.array([[0,0], [0,1], [1,1], [1,0]]) * 10
sparkle = np.array([[1, 1], [10, 0], [1, -1], [0, -10],
                    [-1, -1], [-10, 0], [-1, 1], [0, 10]])
fork = np.array([[2, 10], [0, -5], [-2, 10], [-2, -10], [2, -10]])

polygons = [
    # square
    generate_regular_polygon(4, radius=1) * 10,
    # decagon
    generate_regular_polygon(10, radius=1) * 10 + np.array([[25, 0]]),
    # triangle
    generate_regular_polygon(3, radius=1) * 10 + np.array([[0, 25]]),
    # two sharp prongs
    fork + np.array([[25, 25]]),
    # a four-sided star
    sparkle + np.array([[50, 0]]),
    # same star, but winding in the opposite direction
    sparkle[::-1] + np.array([[50, 26]]),
    # problem shape —
    # lighting bolt with sharp angles and overlapping edge widths
    np.array([[10.97627008, 14.30378733],
              [12.05526752, 10.89766366],
              [8.47309599, 12.91788226],
              [8.75174423, 17.83546002],
              [19.27325521, 7.66883038],
              [15.83450076, 10.5778984]],
             ) + np.array([[60, -15]]),
    ]

paths = [
    # a simple backwards-c shape
    path + np.array([[0, 50]]),
    # an unclosed decagon
    generate_regular_polygon(10, radius=1) * 10 + np.array([[25, 50]]),
    # a three-point straight line (tests collinear points in path)
    np.array([[0, -10], [0, 0], [0, 10]]) + np.array([[50, 50]]),
    ]

shapes = polygons + paths
shape_types=['polygon'] * len(polygons) + ['path'] * len(paths)

viewer = napari.Viewer()
shapes_layer = viewer.add_shapes(shapes, shape_type=shape_types, name='shapes')

add_helper_layers(viewer, source_layer=shapes_layer)
viewer.layers.selection = {shapes_layer}
viewer.reset_view()

if __name__ == '__main__':
    napari.run()
