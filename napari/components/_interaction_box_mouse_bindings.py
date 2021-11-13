import numpy as np

from ..utils.transforms import Affine
from ._interaction_box_constants import Box


def inside_boxes(boxes):
    """Checks which boxes contain the origin. Boxes need not be axis aligned

    Parameters
    ----------
    boxes : (N, 8, 2) array
        Array of N boxes that should be checked

    Returns
    -------
    inside : (N,) array of bool
        True if corresponding box contains the origin.
    """

    AB = boxes[:, 0] - boxes[:, 6]
    AM = boxes[:, 0]
    BC = boxes[:, 6] - boxes[:, 4]
    BM = boxes[:, 6]

    ABAM = np.multiply(AB, AM).sum(1)
    ABAB = np.multiply(AB, AB).sum(1)
    BCBM = np.multiply(BC, BM).sum(1)
    BCBC = np.multiply(BC, BC).sum(1)

    c1 = 0 <= ABAM
    c2 = ABAM <= ABAB
    c3 = 0 <= BCBM
    c4 = BCBM <= BCBC

    inside = np.all(np.array([c1, c2, c3, c4]), axis=0)

    return inside


class InteractionBoxMouseBindings:
    def __init__(self, viewer, interaction_box_visual):
        self._selected_vertex: int = None
        self._fixed_vertex: int = None
        self._fixed_aspect: float = None
        self._viewer = viewer
        self._interaction_box_model = viewer.overlays.interaction_box
        self._interaction_box_visual = interaction_box_visual
        viewer.layers.events.inserted.connect(self._on_add_layer)
        self.initialize_mouse_events(viewer)

    def _on_add_layer(self, event):
        """Gets called when layer is added and adds event listener to mdoe change"""
        layer = event.value
        layer.events.mode.connect(self._on_mode_change)

    def _on_mode_change(self, event):

        viewer = self._viewer
        if event.mode == 'transform':
            viewer.layers.selection.active = event.source
            viewer.overlays.interaction_box.points = (
                viewer.layers.selection.active.extent.data
            )
            viewer.overlays.interaction_box.transform = event.source.affine

            viewer.overlays.interaction_box.show = True
            viewer.overlays.interaction_box.show_vertices = True
            viewer.overlays.interaction_box.show_handle = True
            viewer.overlays.interaction_box.allow_new_selection = False

            viewer.overlays.interaction_box.events.transform_drag.connect(
                self._on_tranform_change
            )
        else:
            viewer.overlays.interaction_box.show = False
            viewer.overlays.interaction_box.points = None
            viewer.overlays.interaction_box.transform = Affine()

    def _on_tranform_change(self, event):
        self._viewer.layers.selection.active.affine = event.value

    def initialize_mouse_events(self, viewer):
        """Adds event handling functions to the layer"""

        @viewer.mouse_move_callbacks.append
        def mouse_move(viewer, event):
            if (
                not self._interaction_box_model.show
                or self._interaction_box_model._box is None
            ):
                return

            # The _box of the visual model has the handle
            box = self._interaction_box_visual._box
            coord = event.position
            distances = abs(box - coord)

            # Get the vertex sizes
            sizes = (
                self._interaction_box_visual._vertex_size / 2
            ) / self._viewer.camera.zoom

            # Check if any matching vertices
            matches = np.all(distances <= sizes, axis=1).nonzero()
            if len(matches[0]) > 0:
                self._selected_vertex = matches[0][-1]
                # Exclde center vertex
                if self._selected_vertex == Box.CENTER:
                    self._selected_vertex = None
            else:
                self._selected_vertex = None

        @viewer.mouse_drag_callbacks.append
        def mouse_drag(viewer, event):
            if not self._interaction_box_model.show:
                return

            # Handling drag start, decide what action to take
            self._set_drag_start_values(viewer, event.position)
            drag_callback = None
            final_callback = None
            if self._selected_vertex is not None:
                if self._selected_vertex == Box.HANDLE:
                    drag_callback = self._on_drag_rotation
                    final_callback = self._on_final_transform
                    yield
                else:
                    self._fixed_vertex = (
                        self._selected_vertex + 4
                    ) % Box.LEN_WITHOUT_HANDLE
                    drag_callback = self._on_drag_scale
                    final_callback = self._on_final_transform
                    yield
            else:
                if (
                    self._interaction_box_model._box is not None
                    and self._interaction_box_model.show
                    and inside_boxes(
                        np.array(
                            [
                                self._interaction_box_model._box
                                - self._drag_start_coordinates
                            ]
                        )
                    )[0]
                ):
                    drag_callback = self._on_drag_translate
                    final_callback = self._on_final_transform

                    yield
                else:
                    if self._interaction_box_model.allow_new_selection:
                        self._interaction_box_model.points = None
                        self._interaction_box_model.transform = Affine()
                        drag_callback = self._on_drag_newbox
                        final_callback = self._on_end_newbox
                    yield
            # Handle events during dragging
            while event.type == 'mouse_move':
                if drag_callback is not None:
                    drag_callback(viewer, event)
                yield

            if final_callback is not None:
                final_callback(viewer, event)

            self._clear_drag_start_values()

    def _set_drag_start_values(self, viewer, position):
        """Gets called whenever a drag is started to remember starting values"""

        self._drag_start_coordinates = np.array(position)
        self._drag_start_box = np.copy(self._interaction_box_visual._box)
        self._interaction_box_model.transform_start = (
            self._interaction_box_model.transform
        )

    def _clear_drag_start_values(self):
        """Gets called at the end of a drag to reset remembered values"""

        self._drag_start_coordinates = None
        self._drag_start_box = None

    def _on_drag_rotation(self, viewer, event):
        """Gets called upon mouse_move in the case of a rotation"""
        center = self._drag_start_box[Box.CENTER]
        handle = self._drag_start_box[Box.HANDLE]
        mouse_offset = np.array(event.position) - center
        handle_offset = handle - center
        angle = np.degrees(
            np.arctan2(mouse_offset[0], mouse_offset[1])
            - np.arctan2(handle_offset[0], handle_offset[1])
        )

        if np.linalg.norm(mouse_offset) < 1:
            angle = 0
        elif self._fixed_aspect:
            angle = np.round(angle / 45) * 45

        tform1 = Affine(translate=-center)
        tform2 = Affine(rotate=-angle)
        tform3 = Affine(translate=center)
        transform = (
            tform3.compose(tform2)
            .compose(tform1)
            .compose(self._interaction_box_model.transform_start)
        )
        self._interaction_box_model.transform = transform
        self._interaction_box_model.transform_drag = transform

    def _on_drag_scale(self, viewer, event):
        """Gets called upon mouse_move in the case of a scaling operation"""

        # Transform everything back into axis-aligned space with fixed point at origin
        transform = self._interaction_box_model.transform_start.inverse
        center = transform(self._drag_start_box[self._fixed_vertex])
        transform = Affine(translate=-center).compose(transform)
        coord = transform(np.array(event.position))
        drag_start = transform(self._drag_start_box[self._selected_vertex])
        # If sidepoint of fixed aspect ratio project offset onto vector along which to scale
        # Since the fixed verted is now at the origin this vector is drag_start
        if self._fixed_aspect or self._selected_vertex % 2 == 1:
            offset = coord - drag_start
            offset_proj = (
                np.dot(drag_start, offset) / (np.linalg.norm(drag_start) ** 2)
            ) * drag_start

            # Prevent numeric instabilities
            offset_proj[np.abs(offset_proj) < 1e-5] = 0
            drag_start[drag_start == 0.0] = 1e-5

            scale = np.array([1.0, 1.0]) + (offset_proj) / drag_start
        else:
            scale = coord / drag_start

        # Apply scaling
        transform = Affine(scale=scale).compose(transform)

        # translate back and apply intial rotation again
        transform = Affine(translate=center).compose(transform)
        transform = self._interaction_box_model.transform_start.compose(
            transform
        )
        # Chain with original transform
        transform = transform.compose(
            self._interaction_box_model.transform_start
        )

        self._interaction_box_model.transform = transform
        self._interaction_box_model.transform_drag = transform

    def _on_drag_translate(self, viewer, event):
        """Gets called upon mouse_move in the case of a translation operation"""

        offset = np.array(event.position) - self._drag_start_coordinates

        transform = Affine(translate=offset).compose(
            self._interaction_box_model.transform_start
        )
        self._interaction_box_model.transform = transform
        self._interaction_box_model.transform_drag = transform

    def _on_final_transform(self, viewer, event):
        """Gets called upon mouse_move in the case of a translation operation"""
        self._interaction_box_model.transform_final = (
            self._interaction_box_model.transform
        )

    def _on_drag_newbox(self, viewer, event):
        """Gets called upon mouse_move in the case of a drawing a new box"""

        self._interaction_box_model.points = np.array(
            [
                self._drag_start_coordinates,
                np.array(event.position),
            ]
        )
        self._interaction_box_model.show = True
        self._interaction_box_model.show_handle = False
        self._interaction_box_model.show_vertices = False
        self._interaction_box_model.selection_box_drag = (
            self._interaction_box_model._box[Box.WITHOUT_HANDLE]
        )

    def _on_end_newbox(self, viewer, event):
        """Gets called when dragging ends in the case of a drawing a new box"""

        if self._interaction_box_model._box is not None:
            self._interaction_box_model.selection_box_final = (
                self._interaction_box_model._box[Box.WITHOUT_HANDLE]
            )
