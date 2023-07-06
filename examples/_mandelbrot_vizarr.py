import logging
import sys
import warnings

import napari
from napari import Viewer
from napari.experimental._progressive_loading import \
    add_progressive_loading_image
from napari.experimental._progressive_loading_datasets import \
    mandelbrot_dataset

warnings.simplefilter(action='ignore', category=FutureWarning)


LOGGER = logging.getLogger("mandelbrot_vizarr")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)


if __name__ == "__main__":
    import yappi

    def start_yappi():
        """Start yappi."""
        yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time
        yappi.start()

    # This dataset is 2D and visualized in 2D
    ndisplay = 2
    large_image = mandelbrot_dataset(max_levels=21)

    # This dataset is 3D and visualized in 2D
    # from napari.experimental._progressive_loading_datasets import openorganelle_mouse_kidney_em
    # large_image = openorganelle_mouse_kidney_em()

    # This dataset is 3D and visualized in 3D
    # ndisplay = 3
    # large_image = mandelbulb_dataset(max_levels=3)

    global viewer
    viewer = Viewer(ndisplay=ndisplay)
    
    multiscale_img = large_image["arrays"]
    viewer._layer_slicer._force_sync = False

    rendering_mode = "progressive_loading"

    if rendering_mode == "progressive_loading":
        # Make an object that creates/manages all scale nodes
        add_progressive_loading_image(
            multiscale_img,
            viewer=viewer,
            contrast_limits=[0, 255],
            colormap='PiYG',
            ndisplay=ndisplay,
        )
    else:
        layer = viewer.add_image(multiscale_img)

    def stop_yappi():
        """Stop yappi."""
        yappi.stop()

        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()

    napari.run()

    def yappi_stats():
        """Save yappi stats."""
        import time

        timestamp = time.time()

        filename = f"/tmp/mandelbrot_vizarr_{timestamp}.prof"

        func_stats = yappi.get_func_stats()
        func_stats.save(filename, type='pstat')
