def dims_scroll(viewer, event):
    """Scroll the dimensions slider."""
    if 'Control' not in event.modifiers:
        return
    if event.native.inverted():
        viewer.dims._scroll_progress += event.delta[1]
    else:
        viewer.dims._scroll_progress -= event.delta[1]
    while abs(viewer.dims._scroll_progress) >= 1:
        if viewer.dims._scroll_progress < 0:
            viewer.dims._increment_dims_left()
            viewer.dims._scroll_progress += 1
        else:
            viewer.dims._increment_dims_right()
            viewer.dims._scroll_progress -= 1


def update_measure(viewer, event):
    print(event.position)
