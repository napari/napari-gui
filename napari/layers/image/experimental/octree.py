"""Octree class.
"""
from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi

from ....types import ArrayLike

TileArray = List[List[np.ndarray]]
Levels = List[TileArray]


class ChunkData:
    """One chunk of the full image.

    A chunk is a 2D tile or a 3D sub-volume.

    Parameters
    ----------
    data : ArrayLike
        The data to draw for this chunk.
    pos : Tuple[float, float]
        The x, y coordinates of the chunk.
    size : float
        The size of the chunk, the chunk is square/cubic.
    """

    def __init__(self, data: ArrayLike, pos: Tuple[float, float], size: float):
        self.data = data
        self.pos = pos
        self.size = size


def _create_tiles(array: np.ndarray, tile_size: int) -> np.ndarray:
    """
    Return an NxM array of (tile_size, tile_size) ndarrays except the edge
    tiles might be smaller if the array did not divide evenly.

    TODO_OCTREE: slices_from_chunks from dask.array.core possibly does
    the same thing, if we are going to use this in production.

    Parameters
    ----------
    array : np.ndarray
        The array to create tiles out of.
    tiles_size : int
        Edge length of the square tiles.
    """
    if array.ndim != 3:
        raise ValueError(f"Unexpected array dimension {array.ndim}")
    (rows, cols, _) = array.shape

    tiles = []

    print(f"_create_tiles array={array.shape} tile_size={tile_size}")

    row = 0
    while row < rows:
        row_tiles = []
        col = 0
        while col < cols:
            tile = array[row : row + tile_size, col : col + tile_size, :]
            row_tiles.append(tile)
            col += tile_size
        tiles.append(row_tiles)
        row += tile_size

    return tiles


def _create_downsampled_tile(
    ul: np.ndarray, ur: np.ndarray, ll: np.ndarray, lr: np.ndarray
) -> np.ndarray:
    """Create one parent tile from four child tiles.

    Parameters
    ----------
    ul : np.ndarray
        Upper left child tile.
    ur : np.ndarray
        Upper right child tile.
    ll : np.ndarray
        Lower left child tile.
    lr : np.ndarray
        Lower right child tile.
    """
    row1 = np.hstack((ul, ur))
    row2 = np.hstack((ll, lr))
    combined_tile = np.vstack((row1, row2))

    # Down sample by half.
    return ndi.zoom(combined_tile, [0.5, 0.5, 1])


def _create_coarser_level(tiles: TileArray) -> TileArray:
    """Return the next coarser level of tiles.

    Combine each 2x2 group of tiles into one downsampled tile.

    tiles : TileArray
        The tiles to combine.
    """

    new_tiles = []

    for row in range(0, len(tiles), 2):
        row_tiles = []
        for col in range(0, len(tiles[row]), 2):
            tile = _create_downsampled_tile(
                tiles[row][col],
                tiles[row][col + 1],
                tiles[row + 1][col],
                tiles[row + 1][col + 1],
            )
            row_tiles.append(tile)
        new_tiles.append(row_tiles)

    return new_tiles


def _print_level_tiles(tiles):
    """Print information about these tiles.
    """
    num_rows = len(tiles)

    num_cols = None
    for row in tiles:
        if num_cols is None:
            num_cols = len(row)
        else:
            assert num_cols == len(row)

    print(f"{num_rows} x {num_cols} = {num_rows * num_cols}")

    for row in tiles:
        for tile in row:
            pass  # print(tile.shape)


def _print_levels(levels: Levels):
    """Print information about the levels.

    Parameters
    ----------
    levels : Levels
        Print information about these levels.
    """
    print(f"{len(levels)} levels:")
    for level in levels:
        _print_level_tiles(level)


def _print_tiles(node, level=0):
    assert node is not None
    assert node.tile is not None
    node.print_info(level)
    for child in node.children:
        if child is not None:
            _print_tiles(child, level + 1)


class OctreeLevel:
    """One level of the octree.

    A level contains a 2D or 3D array of tiles.
    """

    def __init__(self, level_index: int, tiles: TileArray):
        self.level_index = level_index
        self.tiles = tiles

    def print_info(self):
        """Print information about this level."""
        nrows = len(self.tiles)
        ncols = len(self.tiles[0])
        print(f"level={self.level_index} dim={nrows}x{ncols}")

    @property
    def chunks(self):
        nrows = len(self.tiles)
        chunks = []
        x = 0
        y = 0
        size = 1 / nrows
        for row in self.tiles:
            x = 0
            for tile in row:
                chunks.append(ChunkData(tile, [x, y], size))
                x += size
            y += size
        return chunks


class Octree:
    """A region octree to hold images.

    Today the octree is full/complete meaning every node has 4 or 8
    children, and every leaf node is at the same level of the tree. This
    makes sense for region/image trees, because the image exists
    everywhere.

    Since we are a complete tree we don't need actual nodes with references
    to the node's children. Instead, every level is just an array, and
    going from parent to child or child to parent is trivial, you just
    need to of double or half the indexes.


    Future Work
    -----------
    Support geometry, like points and meshes, not just images. For geometry
    a sparse octree might make more sense. With geometry there might be
    lots of empty space in between small dense pockets of geometry. Some
    parts of tree might need to be very deep, but it would be a waste to be
    that deep everywhere.

    Parameters
    ----------
    levels : Levels
        All the levels of the tree
    """

    def __init__(self, levels: Levels):
        self.levels = [
            OctreeLevel(i, level) for (i, level) in enumerate(levels)
        ]
        self.num_levels = len(self.levels)

    def print_info(self):
        """Print information about our tiles."""
        for level in self.levels:
            level.print_info()

    @classmethod
    def from_image(cls, image: np.ndarray):
        """Create octree from given single image.

        Parameters
        ----------
        image : ndarray
            Create the octree for this single image.
        """
        TILE_SIZE = 64
        tiles = _create_tiles(image, TILE_SIZE)
        levels = [tiles]

        # Keep combining tiles until there is one root tile.
        while len(levels[-1]) > 1:
            next_level = _create_coarser_level(levels[-1])
            levels.append(next_level)

        # _print_levels(levels)

        return Octree(levels)


if __name__ == "__main__":
    tree = Octree()
