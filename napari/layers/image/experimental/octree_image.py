"""OctreeImage class.
"""
from typing import List

import numpy as np

from ....components.experimental.chunk import (
    ChunkKey,
    ChunkRequest,
    chunk_loader,
)
from ....utils.events import Event
from ..image import Image
from ._chunked_slice_data import ChunkedSliceData
from ._octree_multiscale_slice import OctreeMultiscaleSlice
from .octree_intersection import OctreeIntersection
from .octree_level import OctreeLevelInfo
from .octree_util import ChunkData, ImageConfig

DEFAULT_TILE_SIZE = 64


class OctreeImage(Image):
    """OctreeImage layer.

    Experimental variant of Image that renders using an Octree.

    Intended to eventually replace Image.
    """

    def __init__(self, *args, **kwargs):
        self.count = 0
        self._tile_size = DEFAULT_TILE_SIZE

        # Is this the same as Image._data_level? Which should we use?
        self._octree_level = None

        self._corners_2d = None
        self._auto_level = True
        self._track_view = True
        self._slice = None

        self.show_grid = True  # Get/set directly.

        super().__init__(*args, **kwargs)
        self.events.add(auto_level=Event, octree_level=Event, tile_size=Event)

    def _get_value(self):
        """Override Image._get_value()."""
        return (0, (0, 0))  # Fake for now until have octree version.

    @property
    def loaded(self):
        """Has the data for this layer been loaded yet."""
        # TODO_OCTREE: what here?
        return True

    @property
    def _empty(self) -> bool:
        return False  # TODO_OCTREE: what here?

    def _update_thumbnail(self):
        # TODO_OCTREE: replace Image._update_thumbnail with nothing for
        # the moment until we decide how to do thumbnail.
        pass

    @property
    def _data_view(self):
        """Viewable image for the current slice. (compatibility)"""
        # Override Image._data_view
        return np.zeros((64, 64, 3))  # fake: does octree need this?

    @property
    def track_view(self) -> bool:
        """Return True if we changing what's dispays as the view changes.

        Return
        ------
        bool
            True if we are tracking the current view.
        """
        return self._track_view

    @track_view.setter
    def track_view(self, value: bool) -> None:
        """Set whether we are tracking the current view.

        Parameters
        ----------
        value : bool
            True if we should track the current view.
        """
        self._track_view = value

    @property
    def tile_size(self) -> int:
        """Return the edge length of single tile, for example 256.

        Return
        ------
        int
            The edge length of a single tile.
        """
        return self._tile_size

    @property
    def tile_shape(self) -> tuple:
        """Return the shape of a single tile, for example 256x256x3.

        Return
        ------
        tuple
            The shape of a single tile.
        """
        # TODO_OCTREE: Must be an easier way to get this shape based on
        # information already stored in Image class?
        if self.multiscale:
            init_shape = self.data[0].shape
        else:
            init_shape = self.data.shape

        tile_shape = (self.tile_size, self.tile_size)

        if self.rgb:
            # Add the color dimension (usually 3 or 4)
            tile_shape += (init_shape[-1],)

        return tile_shape

    @tile_size.setter
    def tile_size(self, tile_size: int) -> None:
        self._tile_size = tile_size
        self.events.tile_size()
        self._slice = None
        self.refresh()

    @property
    def image_config(self) -> ImageConfig:
        """Return information about the current octree.

        Return
        ------
        ImageConfig
            Basic image configuration.
        """
        if self._slice is None:
            return None
        return self._slice.image_config

    @property
    def octree_level_info(self) -> OctreeLevelInfo:
        """Return information about the current level of the current octree.

        Returns
        -------
        OctreeLevelInfo
            Information about the current octree level.
        """
        if self._slice is None:
            return None
        return self._slice.octree_level_info

    @property
    def auto_level(self) -> bool:
        """Return True if we are computing the octree level automatically.

        When viewing the octree normally, auto_level is always True, but
        during debugging or other special situations it might be off.

        Returns
        -------
        bool
            True if we are computing the octree level automatically.
        """
        return self._auto_level

    @auto_level.setter
    def auto_level(self, value: bool) -> None:
        """Set whether we are choosing the octree level automatically.

        Parameters
        ----------
        value : bool
            True if we should determine the octree level automatically.
        """
        self._auto_level = value
        self.events.auto_level()

    @property
    def octree_level(self):
        """Return the currently displayed octree level."""
        return self._octree_level

    @octree_level.setter
    def octree_level(self, level: int):
        """Set the octree level we should be displaying.

        Parameters
        ----------
        level : int
            Display this octree level.
        """
        assert 0 <= level < self.num_octree_levels
        self._octree_level = level
        self.events.octree_level()
        self.refresh()  # Create new slice with this level.

    @property
    def num_octree_levels(self) -> int:
        """Return the total number of octree levels."""
        return len(self.data) - 1  # Multiscale

    def _new_empty_slice(self) -> None:
        """Initialize the current slice to an empty image.

        Overides Image._new_empty_slice() and does nothing because we don't
        need an empty slice. We create self._slice when
        self._set_view_slice() is called.

        The empty slice was needed to satisfy the old VispyImageLayer that
        used a single ImageVisual. But OctreeImage is drawn with
        VispyTiledImageVisual. It does not need an empty image. It gets
        chunks from our self.visible_chunks property, and it will just draw
        nothing if that returns an empty list.

        When OctreeImage become the only image class, this can go away.
        """

    @property
    def visible_chunks(self) -> List[ChunkData]:
        """Chunks in the current slice which in currently in view."""
        # This will be None if we have not been drawn yet.
        if self._corners_2d is None:
            return []

        auto_level = self.auto_level and self.track_view

        if self._slice is None:
            return
        chunks = self._slice.get_visible_chunks(self._corners_2d, auto_level)

        # If we switched to a new octree level, update our currently shown level.
        slice_level = self._slice.octree_level
        if self._octree_level != slice_level:
            self._octree_level = slice_level
            self.events.octree_level()

        # Visible chunks are ones that are already loaded or that we are
        # able to load synchronously. Perhaps in cache, etc.
        # visible_chunks = [
        #    chunk_data
        #    for chunk_data in chunks
        #    if not chunk_data.needs_load or self._load_chunk(chunk_data)
        # ]
        visible_chunks = []
        for chunk_data in chunks:
            if chunk_data.needs_load:
                if self._load_chunk(chunk_data):
                    print("SYNC LOAD")
                    visible_chunks.append(chunk_data)
                else:
                    print("ASYNC LOAD")
            else:
                print("ALREADY LOADED")
                visible_chunks.append(chunk_data)

        return visible_chunks

    def _load_chunk(self, chunk_data: ChunkData) -> None:

        indices = np.array(self._slice_indices)
        key = ChunkKey(self, indices, chunk_data.location)

        chunks = {'data': chunk_data.data}

        chunk_data.loading = True

        # Create the ChunkRequest and load it with the ChunkLoader.
        request = chunk_loader.create_request(self, key, chunks)

        satisfied_request = chunk_loader.load_chunk(request)

        if satisfied_request is None:
            return False  # Load was async.

        # Load was sync so we have the data already, we can assign it
        # here and return this as a visible chunk immediately.
        chunk_data.data = satisfied_request.chunks.get('data')
        print("SYNC LOAD")
        return True

    def _on_data_loaded(self, data: ChunkedSliceData, sync: bool) -> None:
        """The given data a was loaded, use it now."""

    def _update_draw(self, scale_factor, corner_pixels, shape_threshold):

        # Need refresh if have not been draw at all yet.
        need_refresh = self._corners_2d is None

        # Compute self._corners_2d which we use for intersections.
        data_corners = self._transforms[1:].simplified.inverse(corner_pixels)
        self._corners_2d = self._convert_to_corners_2d(data_corners)

        super()._update_draw(scale_factor, corner_pixels, shape_threshold)

        if need_refresh:
            self.refresh()

    def get_intersection(self) -> OctreeIntersection:
        """The the interesection between the current view and the octree.

        Returns
        -------
        OctreeIntersection
            The intersection between the current view and the octree.
        """
        if self._slice is None:
            return None

        return self._slice.get_intersection(self._corners_2d, self.auto_level)

    def _convert_to_corners_2d(self, data_corners):
        """
        Get data corners in 2d.
        """
        # TODO_OCTREE: This is placeholder. Need to handle dims correctly.
        if self.ndim == 2:
            return data_corners
        return data_corners[:, 1:3]

    def _outside_data_range(self, indices) -> bool:
        """Return True if requested slice is outside of data range.

        Return
        ------
        bool
            True if requested slice is outside data range.
        """

        extent = self._extent_data
        not_disp = self._dims.not_displayed

        return np.any(
            np.less(
                [indices[ax] for ax in not_disp],
                [extent[0, ax] for ax in not_disp],
            )
        ) or np.any(
            np.greater(
                [indices[ax] for ax in not_disp],
                [extent[1, ax] for ax in not_disp],
            )
        )

    def _set_view_slice(self):
        """Set the view given the indices to slice with.

        This replaces Image._set_view_slice() entirely. The hope is eventually
        this class OctreeImage becomes Image. And the non-tiled multiscale
        logic in Image._set_view_slice goes away entirely.
        """
        if self._slice is not None:  # bail as a test
            return
        indices = np.array(self._slice_indices)
        if self._outside_data_range(indices):
            return
        delay_ms = 250
        image_config = ImageConfig.create(
            self.data[0].shape, self._tile_size, delay_ms
        )

        self._slice = OctreeMultiscaleSlice(
            self.data, image_config, self._raw_to_displayed
        )

    def on_chunk_loaded(self, request: ChunkRequest) -> None:
        """An asynchronous ChunkRequest was loaded.

        Override Image.on_chunk_loaded() fully.

        Parameters
        ----------
        request : ChunkRequest
            This request was loaded.
        """
        self._slice.on_chunk_loaded(request)

    def refresh(self, event=None):
        if self.count < 10:  # as a test
            self.count += 1
            super().refresh(event)
