import numpy as np

from napari._vispy.overlays.slice_text import VispySliceTextOverlay
from napari.components.overlays.slice_text import SliceTextOverlay


def test_slice_text_parse_function_update(make_napari_viewer):
    viewer = make_napari_viewer()

    viewer.add_image(np.zeros((15, 15, 15)))
    viewer.slice_text.visible = True

    model = SliceTextOverlay()
    visual = VispySliceTextOverlay(overlay=model, viewer=viewer)

    assert visual.node.text.text == f'{model.text_prefix}0={15 // 2}\n'

    model.slice_parse_function = (
        lambda viewer, dim_idx: f'{type(viewer).__name__} {dim_idx}\n'
    )

    assert visual.node.text.text == f'{model.text_prefix}Viewer 0\n'
