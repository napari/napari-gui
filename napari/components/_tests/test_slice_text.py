from napari.components.overlays.slice_text import SliceTextOverlay


def test_slice_text():
    """Test creating scale bar object"""
    scale_bar = SliceTextOverlay()
    assert scale_bar is not None
