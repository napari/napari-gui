"""Check the triangulation of a complex polygon with a hole in it.

This PR uses the triangle library to display a complex polygon in both
matplotlib and napari.

See issue https://github.com/napari/napari/issues/5673 and PR
https://github.com/napari/napari/pull/6654
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

try:
    from triangle import triangulate
except ModuleNotFoundError:
    from vispy.geometry.triangulation import Triangulation
    def triangulate(params, opts=None):
        vertices_raw = params['vertices']
        edges = params['segments']
        tri = Triangulation(vertices_raw, edges)
        tri.triangulate()
        return {'vertices': tri.pts, 'triangles': tri.tris}

import napari
from napari.layers.shapes import _shapes_utils


def pltri(vertices, triangles, *, mask=None, ax=None):
    """Plot a triangulation in Matplotlib.

    This function inverts the y-axis and transposes x and y to match
    napari's row/columns coordinates. (Which also matches Matplotlib's
    coordinates for images. ;)
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.triplot(vertices[:, 1], -vertices[:, 0], triangles)
    if mask is not None:
        centers = np.mean(vertices[triangles], axis=1)
        color = np.where(mask, 'blue', 'red')
        ax.scatter(centers[:, 1], -centers[:, 0], color=color)


# polygon with hole from https://github.com/napari/napari/issues/5673
# rounded to two decimal digits.
a = np.array(
    [[-28.58, 196.34], [-28.08, 196.82], [-28.36, 197.22], [-28.78, 197.39],
     [-28.86, 197.84], [-29.05, 198.46], [-28.97, 199.  ], [-28.46, 199.89],
     [-24.77, 199.9 ], [-24.92, 200.17], [-25.87, 200.76], [-26.48, 200.67],
     [-26.83, 200.89], [-26.73, 201.61], [-26.28, 202.11], [-25.98, 202.58],
     [-25.5 , 202.82], [-25.27, 203.31], [-25.39, 203.73], [-25.67, 204.21],
     [-25.72, 205.03], [-25.49, 205.66], [-25.17, 205.77], [-24.7 , 205.94],
     [-24.62, 206.49], [-24.24, 206.79], [-23.57, 207.12], [-22.83, 208.02],
     [-22.09, 209.43], [-22.1 , 209.84], [-22.27, 210.32], [-22.15, 210.66],
     [-22.25, 211.19], [-23.66, 211.67], [-24.37, 211.93], [-25.48, 211.75],
     [-25.84, 211.84], [-25.66, 211.33], [-25.73, 211.04], [-26.02, 210.95],
     [-26.4 , 210.68], [-26.74, 210.69], [-27.29, 211.28], [-27.18, 211.87],
     [-26.73, 212.07], [-26.74, 212.83], [-27.47, 212.58], [-28.3 , 212.46],
     [-28.75, 212.2 ], [-29.26, 211.52], [-29.4 , 211.33], [-29.91, 210.9 ],
     [-30.42, 210.62], [-31.14, 210.06], [-32.17, 208.93], [-32.77, 208.22],
     [-33.23, 207.46], [-33.61, 206.42], [-33.67, 205.91], [-33.94, 205.78],
     [-33.8 , 205.17], [-33.99, 204.68], [-33.79, 203.59], [-33.92, 202.99],
     [-33.86, 202.57], [-34.26, 201.54], [-34.42, 200.69], [-34.8 , 200.07],
     [-34.82, 199.62], [-34.46, 199.19], [-34.44, 198.86], [-34.  , 198.42],
     [-34.14, 198.38], [-33.87, 198.24], [-33.28, 198.25], [-32.61, 197.93],
     [-32.43, 198.25], [-31.66, 198.22], [-30.73, 197.57], [-29.88, 197.06],
     [-29.88, 197.06], [-28.58, 196.34], [-28.96, 208.98], [-28.65, 208.54],
     [-28.85, 208.07], [-29.24, 207.53], [-29.88, 207.  ], [-30.65, 207.75],
     [-30.55, 208.11], [-30.23, 208.29], [-30.07, 208.85], [-29.74, 209.02],
     [-29.26, 209.33], [-28.96, 208.98]]
    )


# First, check the utils code manually using matplotlib.
v, e = _shapes_utils._normalize_vertices_and_edges(a, close=True)
res = triangulate({'vertices': v, 'segments': e}, opts='p')
v2, t = res['vertices'], res['triangles']
centers = np.mean(v2[t], axis=1)
in_poly = measure.points_in_poly(centers, a)

fig, ax = plt.subplots()
pltri(res['vertices'], res['triangles'], mask=in_poly, ax=ax)
fig.show()


# next, draw the shape in napari
viewer = napari.Viewer()
layer = viewer.add_shapes(a, shape_type=['polygon'])

# these settings help to visualise the polygon data directly in the
# shapes layer.
layer.mode = 'direct'
layer.edge_width = 0.1
layer.selected_data = {0}

if __name__ == '__main__':
    napari.run()
