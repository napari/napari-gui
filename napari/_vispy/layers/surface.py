import numpy as np
from vispy.color import Colormap as VispyColormap

from ..visuals.surface import SurfaceVisual
from .base import VispyBaseLayer


class VispySurfaceLayer(VispyBaseLayer):
    """Vispy view for the surface layer.

    View is based on the vispy mesh node and uses default values for
    lighting direction and lighting color. More information can be found
    here https://github.com/vispy/vispy/blob/main/vispy/visuals/mesh.py
    """

    def __init__(self, layer):
        node = SurfaceVisual()
        super().__init__(layer, node)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.layer.events.shading.connect(self._on_shading_change)
        self.layer.events.wireframe.connect(self._on_wireframe_change)
        self.layer.events.face_normals.connect(self._on_face_normals_change)
        self.layer.events.face_normals_color.connect(
            self._on_face_normals_change
        )
        self.layer.events.face_normals_length.connect(
            self._on_face_normals_change
        )
        self.layer.events.face_normals_width.connect(
            self._on_face_normals_change
        )
        self.layer.events.vertex_normals.connect(
            self._on_vertex_normals_change
        )
        self.layer.events.vertex_normals_color.connect(
            self._on_vertex_normals_change
        )
        self.layer.events.vertex_normals_length.connect(
            self._on_vertex_normals_change
        )
        self.layer.events.vertex_normals_width.connect(
            self._on_vertex_normals_change
        )

        self.reset()
        self._on_data_change()

    def _on_data_change(self):
        if len(self.layer._data_view) == 0 or len(self.layer._view_faces) == 0:
            vertices = None
            faces = None
            vertex_values = np.array([0])
        else:
            # Offsetting so pixels now centered
            # coerce to float to solve vispy/vispy#2007
            vertices = np.asarray(
                self.layer._data_view[:, ::-1], dtype=np.float32
            )
            faces = self.layer._view_faces
            vertex_values = self.layer._view_vertex_values

        if (
            vertices is not None
            and self.layer._ndisplay == 3
            and self.layer.ndim == 2
        ):
            vertices = np.pad(vertices, ((0, 0), (0, 1)))

        # manually detach filters when we go to 2D to avoid dimensionality issues
        # see comments in napari#3475. The filter is set again after set_data!
        if self.layer._ndisplay == 2:
            filt = self.node.shading_filter
            try:
                self.node.detach(filt)
                self.node.shading = None
                self.node.shading_filter = None
            except ValueError:
                # sometimes we try to detach non-attached filters, which causes a ValueError
                pass

        self.node.set_data(
            vertices=vertices, faces=faces, vertex_values=vertex_values
        )
        self.node.face_normals.set_data(self.node.mesh_data)
        self.node.vertex_normals.set_data(self.node.mesh_data)
        self._on_shading_change()

        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_colormap_change(self):
        if self.layer.gamma != 1:
            # when gamma!=1, we instantiate a new colormap with 256 control
            # points from 0-1
            colors = self.layer.colormap.map(
                np.linspace(0, 1, 256) ** self.layer.gamma
            )
            cmap = VispyColormap(colors)
        else:
            cmap = VispyColormap(*self.layer.colormap)
        if self.layer._ndisplay == 3:
            self.node.view_program['texture2D_LUT'] = (
                cmap.texture_lut() if (hasattr(cmap, 'texture_lut')) else None
            )
        self.node.cmap = cmap

    def _on_contrast_limits_change(self):
        self.node.clim = self.layer.contrast_limits

    def _on_gamma_change(self):
        self._on_colormap_change()

    def _on_shading_change(self):
        shading = None if self.layer.shading == 'none' else self.layer.shading
        if self.layer._ndisplay == 3:
            self.node.shading = shading
        self.node.update()

    def _on_wireframe_change(self):
        self.node.wireframe_filter.enabled = self.layer.wireframe
        self.node.update()

    def _on_face_normals_change(self):
        self.node.face_normals.visible = self.layer.face_normals
        if self.node.face_normals.visible:
            self.node.face_normals.set_data(
                length=self.layer.face_normals_length,
                color=self.layer.face_normals_color,
                width=self.layer.face_normals_width,
            )

    def _on_vertex_normals_change(self):
        self.node.vertex_normals.visible = self.layer.vertex_normals
        if self.node.vertex_normals.visible:
            self.node.vertex_normals.set_data(
                length=self.layer.vertex_normals_length,
                color=self.layer.vertex_normals_color,
                width=self.layer.vertex_normals_width,
            )

    def reset(self, event=None):
        super().reset()
        self._on_colormap_change()
        self._on_contrast_limits_change()
        self._on_shading_change()
        self._on_wireframe_change()
        self._on_face_normals_change()
        self._on_vertex_normals_change()
