import numpy as np
from vispy.scene import ArcballCamera, BaseCamera, PanZoomCamera

from napari._vispy.utils.quaternion import quaternion2euler_degrees
from napari.utils.events import disconnect_events


class VispyCamera:
    """Vipsy camera for both 2D and 3D rendering.

    Parameters
    ----------
    view : vispy.scene.widgets.viewbox.ViewBox
        Viewbox for current scene.
    camera : napari.components.Camera
        napari camera model.
    dims : napari.components.Dims
        napari dims model.
    """

    def __init__(self, view, camera, dims) -> None:
        self._view = view
        self._camera = camera
        self._dims = dims

        # Create 2D camera
        self._2D_camera = MouseToggledPanZoomCamera(aspect=1)
        # flip y-axis to have correct alignment
        self._2D_camera.flip = (0, 1, 0)
        self._2D_camera.viewbox_key_event = viewbox_key_event

        # Create 3D camera
        self._3D_camera = MouseToggledArcballCamera(fov=0)
        self._3D_camera.viewbox_key_event = viewbox_key_event

        # Set 2D camera by default
        self._view.camera = self._2D_camera

        self._dims.events.ndisplay.connect(
            self._on_ndisplay_change, position='first'
        )

        self._camera.events.center.connect(self._on_center_change)
        self._camera.events.zoom.connect(self._on_zoom_change)
        self._camera.events.angles.connect(self._on_angles_change)
        self._camera.events.perspective.connect(self._on_perspective_change)
        self._camera.events.mouse_pan.connect(self._on_mouse_toggles_change)
        self._camera.events.mouse_zoom.connect(self._on_mouse_toggles_change)

        self._on_ndisplay_change()

    @property
    def angles(self):
        """3-tuple: Euler angles of camera in 3D viewing, in degrees.
        Note that angles might be different than the ones that might have generated the quaternion.
        """

        if self._view.camera == self._3D_camera:
            # Do conversion from quaternion representation to euler angles
            angles = quaternion2euler_degrees(self._view.camera._quaternion)
        else:
            angles = (0, 0, 90)
        return angles

    @angles.setter
    def angles(self, angles):
        if self.angles == tuple(angles):
            return

        # Only update angles if current camera is 3D camera
        if self._view.camera == self._3D_camera:
            # Create and set quaternion
            quat = self._view.camera._quaternion.create_from_euler_angles(
                *angles,
                degrees=True,
            )
            self._view.camera._quaternion = quat
            self._view.camera.view_changed()

    @property
    def center(self):
        """tuple: Center point of camera view for 2D or 3D viewing."""
        if self._view.camera == self._3D_camera:
            center = tuple(self._view.camera.center)
        else:
            # in 2D, we arbitrarily choose 0.0 as the center in z
            center = (*self._view.camera.center[:2], 0.0)
        # switch from VisPy xyz ordering to NumPy prc ordering
        return center[::-1]

    @center.setter
    def center(self, center):
        if self.center == tuple(center):
            return
        self._view.camera.center = center[::-1]
        self._view.camera.view_changed()

    @property
    def zoom(self):
        """float: Scale from canvas pixels to world pixels."""
        # if self._view.canvas is None:
        #     print()
        canvas_size = np.array(self._view.canvas.size)
        if self._view.camera == self._3D_camera:
            # For fov = 0.0 normalize scale factor by canvas size to get scale factor.
            # Note that the scaling is stored in the `_projection` property of the
            # camera which is updated in vispy here
            # https://github.com/vispy/vispy/blob/v0.6.5/vispy/scene/cameras/perspective.py#L301-L313
            scale = self._view.camera.scale_factor
        else:
            scale = np.array(
                [self._view.camera.rect.width, self._view.camera.rect.height]
            )
            scale[np.isclose(scale, 0)] = 1  # fix for #2875
        zoom = np.min(canvas_size / scale)
        return zoom

    @zoom.setter
    def zoom(self, zoom):
        if self.zoom == zoom:
            return
        scale = np.array(self._view.canvas.size) / zoom
        if self._view.camera == self._3D_camera:
            self._view.camera.scale_factor = np.min(scale)
        else:
            # Set view rectangle, as left, right, width, height
            corner = np.subtract(self._view.camera.center[:2], scale / 2)
            self._view.camera.rect = tuple(corner) + tuple(scale)

    @property
    def perspective(self):
        """Field of view of camera (only visible in 3D mode)."""
        return self._3D_camera.fov

    @perspective.setter
    def perspective(self, perspective):
        if self.perspective == perspective:
            return
        self._3D_camera.fov = perspective
        self._view.camera.view_changed()

    @property
    def mouse_zoom(self) -> bool:
        return self._view.camera.mouse_zoom

    @mouse_zoom.setter
    def mouse_zoom(self, mouse_zoom: bool):
        self._view.camera.mouse_zoom = mouse_zoom

    @property
    def mouse_pan(self) -> bool:
        return self._view.camera.mouse_pan

    @mouse_pan.setter
    def mouse_pan(self, mouse_pan: bool):
        self._view.camera.mouse_pan = mouse_pan

    def _on_ndisplay_change(self):
        if self._dims.ndisplay == 3:
            self._view.camera = self._3D_camera
        else:
            self._view.camera = self._2D_camera

        self._on_mouse_toggles_change()
        self._on_center_change()
        self._on_zoom_change()
        self._on_angles_change()

    def _on_mouse_toggles_change(self):
        self.mouse_pan = self._camera.mouse_pan
        self.mouse_zoom = self._camera.mouse_zoom

    def _on_center_change(self):
        self.center = self._camera.center[-self._dims.ndisplay :]

    def _on_zoom_change(self):
        self.zoom = self._camera.zoom

    def _on_perspective_change(self):
        self.perspective = self._camera.perspective

    def _on_angles_change(self):
        self.angles = self._camera.angles

    def on_draw(self, _event):
        """Called whenever the canvas is drawn.

        Update camera model angles, center, and zoom.
        """
        with self._camera.events.angles.blocker(self._on_angles_change):
            self._camera.angles = self.angles
        with self._camera.events.center.blocker(self._on_center_change):
            self._camera.center = self.center
        with self._camera.events.zoom.blocker(self._on_zoom_change):
            self._camera.zoom = self.zoom
        with self._camera.events.perspective.blocker(
            self._on_perspective_change
        ):
            self._camera.perspective = self.perspective

    def disconnect(self):
        disconnect_events(self._camera.events, self)


def viewbox_key_event(event):
    """ViewBox key event handler.

    Parameters
    ----------
    event : vispy.util.event.Event
        The vispy event that triggered this method.
    """
    return


def add_mouse_pan_zoom_toggles(
    vispy_camera_cls: type[BaseCamera],
) -> type[BaseCamera]:
    """Add separate mouse pan and mouse zoom toggles to VisPy.

    By default, VisPy uses an ``interactive`` toggle that turns *both*
    panning and zooming on and off. This decorator adds separate toggles,
    ``mouse_pan`` and ``mouse_zoom``, to enable controlling them
    separately.

    Parameters
    ----------
    vispy_camera_cls : Type[vispy.scene.cameras.BaseCamera]
        A VisPy camera class to decorate.

    Returns
    -------
    A decorated VisPy camera class.
    """

    class _vispy_camera_cls(vispy_camera_cls):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mouse_pan = True
            self.mouse_zoom = True

        def viewbox_mouse_event(self, event):
            if (
                self.mouse_zoom
                and event.type in ('mouse_wheel', 'gesture_zoom')
                or self.mouse_pan
                and event.type
                in ('mouse_move', 'mouse_press', 'mouse_release')
            ):
                super().viewbox_mouse_event(event)
            else:
                event.handled = False

    return _vispy_camera_cls


MouseToggledPanZoomCamera = add_mouse_pan_zoom_toggles(PanZoomCamera)
MouseToggledArcballCamera = add_mouse_pan_zoom_toggles(ArcballCamera)
