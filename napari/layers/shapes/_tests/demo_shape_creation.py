import argparse
from timeit import default_timer

import numpy as np

import napari


def create_sample_coords(n_polys=3000, n_vertices=16):
    """random circular polygons with given number of vertices"""
    center = np.random.randint(0, 1000, (n_polys, 2))
    radius = (
        1000
        / np.sqrt(n_polys)
        * np.random.uniform(0.9, 1.1, (n_polys, n_vertices))
    )

    phi = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    rays = np.stack([np.sin(phi), np.cos(phi)], 1)

    radius = radius.reshape((-1, n_vertices, 1))
    rays = rays.reshape((1, -1, 2))
    center = center.reshape((-1, 1, 2))
    coords = center + radius * rays
    return coords


def time_me(label, func):
    # print(f'{name} start')
    t = default_timer()
    res = func()
    t = default_timer() - t
    print(f"{label}: {t:.4f} s")
    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-n",
        "--n_polys",
        type=int,
        default=5000,
        help='number of polygons to show',
    )
    parser.add_argument(
        "-t", "--type", type=str, default="path", choices=['path', 'polygon']
    )
    parser.add_argument(
        "-c",
        "--concat",
        action="store_true",
        help='concatenate all coordinates to a single mesh',
    )
    parser.add_argument(
        "-v", "--view", action="store_true", help='show napari viewer'
    )

    args = parser.parse_args()

    coords = create_sample_coords(args.n_polys)

    if args.concat:
        coords = coords.reshape((1, -1, 2))

    print(f'number of polygons: {args.n_polys}')
    print(f'layer type: {args.type}')

    layer = time_me(
        "time to create layer",
        lambda: napari.layers.Shapes(
            coords, shape_type=args.type, edge_color=[1, 0.5, 0.2, 1]
        ),
    )

    if args.view:
        # add the image
        viewer = napari.Viewer()
        viewer.add_layer(layer)
        napari.run()
