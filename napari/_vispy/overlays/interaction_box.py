import numpy as np
from vispy.scene.visuals import Compound, Line, Markers

from napari.utils.transforms.transforms import Affine

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


class VispyInteractionBox:
    def __init__(self, viewer, parent=None, order=0):

        self._viewer = viewer
        self._interaction_box = viewer.overlays.interaction_box
        self.node = Compound([Line(), Markers()], parent=parent)
        self.node.order = order
        self._on_interaction_box_change()
        self.initialize_mouse_events(viewer)
        self._interaction_box.events.points.connect(
            self._on_interaction_box_change
        )
        self._interaction_box.events.show.connect(
            self._on_interaction_box_change
        )
        self._interaction_box.events.show_handle.connect(
            self._on_interaction_box_change
        )
        self._interaction_box.events.show_vertices.connect(
            self._on_interaction_box_change
        )
        self._interaction_box.events.transform.connect(
            self._on_interaction_box_change
        )
        self._highlight_width = 1.5
        self._selected_vertex: int = None
        self._fixed_vertex: int = None
        self._fixed_aspect: float = None
        self._vertex_size = 10
        self._rotation_handle_length = 20
        self._highlight_color = (0, 0.6, 1)
        self._box = None

    @property
    def marker_node(self):
        """sequence of float: Scale factors."""
        return self.node._subvisuals[1]

    @property
    def line_node(self):
        """sequence of float: Scale factors."""
        return self.node._subvisuals[0]

    def _on_interaction_box_change(self, event=None):
        """Called whenever the interaction box changed."""

        # Compute the location and properties of the vertices and box that
        # need to get rendered
        (
            vertices,
            face_color,
            edge_color,
            pos,
            width,
        ) = self._compute_vertices_and_box()

        if vertices is None or len(vertices) == 0:
            vertices = np.zeros((1, self._viewer.dims.ndisplay))
            size = 0
        else:
            vertices = vertices + 0.5
            size = 10

        self.marker_node.set_data(
            vertices,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=1.5,
            symbol='square',
            scaling=False,
        )

        if pos is None or len(pos) == 0:
            pos = np.zeros((1, self._viewer.dims.ndisplay))
            width = 0
        else:
            pos = pos + 0.5
        self.line_node.set_data(pos=pos, color=edge_color, width=width)

    def initialize_mouse_events(self, viewer):
        """Adds event handling functions to the layer"""

        @viewer.mouse_move_callbacks.append
        def mouse_move(viewer, event):
            if (
                not self._interaction_box.show
                or self._interaction_box._box is None
            ):
                return

            box = self._box
            coord = event.position
            distances = abs(box - coord)

            # Get the vertex sizes
            sizes = self._vertex_size / 2

            # Check if any matching vertices
            matches = np.all(distances <= sizes, axis=1).nonzero()
            if len(matches[0]) > 0:
                self._selected_vertex = matches[0][-1]
                # Exclde center vertex
                if self._selected_vertex == Box.CENTER:
                    self._selected_vertex = None
            else:
                self._selected_vertex = None
            # self.events.points_changed()

        @viewer.mouse_drag_callbacks.append
        def mouse_drag(viewer, event):
            if not self._interaction_box.show:
                return

            # Handling drag start, decide what action to take
            self._set_drag_start_values(viewer, event.position)
            drag_callback = None
            final_callback = None
            if self._selected_vertex is not None:
                if self._selected_vertex == Box.HANDLE:
                    drag_callback = self._on_drag_rotation
                    final_callback = self._on_final_tranform
                    yield
                else:
                    self._fixed_vertex = (self._selected_vertex + 4) % Box.LEN
                    drag_callback = self._on_drag_scale
                    final_callback = self._on_final_tranform
                    yield
            else:
                if (
                    self._interaction_box._box is not None
                    and self._interaction_box.show
                    and inside_boxes(
                        np.array(
                            [
                                self._interaction_box._box
                                - self._drag_start_coordinates
                            ]
                        )
                    )[0]
                ):
                    drag_callback = self._on_drag_translate
                    final_callback = self._on_final_tranform

                    yield
                else:
                    if self._interaction_box.allow_new_selection:
                        self._interaction_box.points = None
                        self._interaction_box.transform = Affine()
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
        self._drag_start_box = np.copy(self._box)
        self._interaction_box.transform_start = self._interaction_box.transform

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
            .compose(self._interaction_box.transform_start)
        )
        self._interaction_box.transform = transform
        self._interaction_box.transform_drag = transform

    def _on_drag_scale(self, viewer, event):
        """Gets called upon mouse_move in the case of a scaling operation"""

        # Transform everything back into axis-aligned space with fixed point at origin
        transform = self._interaction_box.transform_start.inverse
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
        transform = self._interaction_box.transform_start.compose(transform)
        # Chain with original transform
        transform = transform.compose(self._interaction_box.transform_start)

        self._interaction_box.transform = transform
        self._interaction_box.transform_drag = transform

    def _on_drag_translate(self, viewer, event):
        """Gets called upon mouse_move in the case of a translation operation"""

        offset = np.array(event.position) - self._drag_start_coordinates

        transform = Affine(translate=offset).compose(
            self._interaction_box.transform_start
        )
        self._interaction_box.transform = transform
        self._interaction_box.transform_drag = transform
        # self.interaction_box.transform_drag = transform

    def _on_final_tranform(self, viewer, event):
        """Gets called upon mouse_move in the case of a translation operation"""
        self._interaction_box.transform_final = self._interaction_box.transform

    def _on_drag_newbox(self, viewer, event):
        """Gets called upon mouse_move in the case of a drawing a new box"""

        self._interaction_box.points = np.array(
            [
                self._drag_start_coordinates,
                np.array(event.position),
            ]
        )
        self._interaction_box.show = True
        self._interaction_box.show_handle = False
        self._interaction_box.show_vertices = False
        self._interaction_box.selection_box_drag = self._interaction_box._box[
            Box.WITHOUT_HANDLE
        ]

    def _on_end_newbox(self, viewer, event):
        """Gets called when dragging ends in the case of a drawing a new box"""
        # self._interaction_box.show = False
        if self._interaction_box._box is not None:
            self._interaction_box.selection_box_final = (
                self._interaction_box._box[Box.WITHOUT_HANDLE]
            )

    def _compute_vertices_and_box(self):
        """Compute location of the box for rendering.

        Returns
        ----------
        vertices : np.ndarray
            Nx2 array of any vertices to be rendered as Markers
        face_color : str
            String of the face color of the Markers
        edge_color : str
            String of the edge color of the Markers and Line for the box
        pos : np.ndarray
            Nx2 array of vertices of the box that will be rendered using a
            Vispy Line
        width : float
            Width of the box edge
        """
        if (
            self._interaction_box._box is not None
            and self._interaction_box.show
        ):
            box = self._interaction_box._box
            if self._interaction_box.show_handle:
                box = self._add_rotation_handle(box)

            if self._selected_vertex is None:
                face_color = 'white'
            else:
                face_color = self._highlight_color

            edge_color = self._highlight_color
            if self._interaction_box.show_vertices:
                if self._interaction_box.show_handle:
                    vertices = box[Box.WITH_HANDLE][:, ::-1]
                else:
                    vertices = box[Box.WITHOUT_HANDLE][:, ::-1]
            else:
                vertices = np.empty((0, 2))

            # Use a subset of the vertices of the interaction_box to plot
            # the line around the edge
            if self._interaction_box.show_handle:
                pos = box[Box.LINE_HANDLE][:, ::-1]
            else:
                pos = box[Box.LINE][:, ::-1]
            width = self._highlight_width
            self._box = box
        else:
            # Otherwise show nothing
            vertices = np.empty((0, 2))
            face_color = 'white'
            edge_color = 'white'
            pos = None
            width = 0
            self._box = None

        return vertices, face_color, edge_color, pos, width

    def _add_rotation_handle(self, box):
        """Adds the rotation handle to the box"""

        if box is not None:
            rot = box[Box.TOP_CENTER]
            length_box = np.linalg.norm(
                box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT]
            )
            if length_box > 0:
                r = self._rotation_handle_length
                rot = (
                    rot
                    - r
                    * (box[Box.BOTTOM_LEFT] - box[Box.TOP_LEFT])
                    / length_box
                )
            box = np.append(box, [rot], axis=0)

        return box
