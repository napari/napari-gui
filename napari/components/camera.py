import warnings

import numpy as np
from scipy.spatial.transform import Rotation as R

from napari._pydantic_compat import validator
from napari.utils.events import EventedModel
from napari.utils.misc import ensure_n_tuple
from napari.utils.translations import trans


class Camera(EventedModel):
    """Camera object modeling position and view of the camera.

    Attributes
    ----------
    center : 3-tuple
        Center of rotation for the camera.
        In 2D viewing the last two values are used.
    zoom : float
        Scale from canvas pixels to world pixels.
    angles : 3-tuple
        Euler angles of camera in 3D viewing (rx, ry, rz), in degrees.
        Only used during 3D viewing.
        Note that Euler angles's intrinsic degeneracy means different
        sets of Euler angles may lead to the same view.
    perspective : float
        Perspective (aka "field of view" in vispy) of the camera (if 3D).
    interactive : bool
        If the camera mouse pan/zoom is enabled or not.
        This attribute is deprecated since 0.5.0 and should not be used.
        Use the mouse_pan and mouse_zoom attributes instead.
    mouse_pan : bool
        If the camera interactive panning with the mouse is enabled or not.
    mouse_zoom : bool
        If the camera interactive zooming with the mouse is enabled or not.
    """

    # fields
    center: tuple[float, float, float] | tuple[float, float] = (
        0.0,
        0.0,
        0.0,
    )
    zoom: float = 1.0
    angles: tuple[float, float, float] = (0.0, 0.0, 90.0)
    perspective: float = 0
    mouse_pan: bool = True
    mouse_zoom: bool = True

    # validators
    @validator('center', 'angles', pre=True, allow_reuse=True)
    def _ensure_3_tuple(cls, v):
        return ensure_n_tuple(v, n=3)

    @property
    def view_direction(self) -> tuple[float, float, float]:
        """3D view direction vector of the camera.

        View direction is calculated from the Euler angles and returned as a
        3-tuple. This direction is in 3D scene coordinates, the world coordinate
        system for three currently displayed dimensions.
        """
        ang = np.deg2rad(self.angles)
        view_direction = (
            # z has a negative sign for the right-handed reference frame
            # flip (#7488)
            -np.sin(ang[2]) * np.cos(ang[1]),
            np.cos(ang[2]) * np.cos(ang[1]),
            -np.sin(ang[1]),
        )
        return view_direction

    @property
    def up_direction(self) -> tuple[float, float, float]:
        """3D direction vector pointing up on the canvas.

        Up direction is calculated from the Euler angles and returned as a
        3-tuple. This direction is in 3D scene coordinates, the world coordinate
        system for three currently displayed dimensions.
        """
        rotation_matrix = R.from_euler(
            seq='yzx', angles=self.angles, degrees=True
        ).as_matrix()
        return (
            # z has a negative sign for the right-handed reference frame
            # flip (#7488)
            -rotation_matrix[2, 2],
            rotation_matrix[1, 2],
            rotation_matrix[0, 2],
        )

    def set_view_direction(
        self,
        view_direction: tuple[float, float, float],
        up_direction: tuple[float, float, float] = (0, -1, 0),
    ):
        """Set camera angles from direction vectors.

        Both the view direction and the up direction are specified in 3D scene
        coordinates, the world coordinate system for three currently displayed
        dimensions.

        The provided up direction must not be parallel to the provided
        view direction. The provided up direction does not need to be orthogonal
        to the view direction. The final up direction will be a vector orthogonal
        to the view direction, aligned with the provided up direction.

        Parameters
        ----------
        view_direction : 3-tuple of float
            The desired view direction vector in 3D scene coordinates, the world
            coordinate system for three currently displayed dimensions.
        up_direction : 3-tuple of float
            A direction vector which will point upwards on the canvas. Defaults
            to (0, -1, 0) unless the view direction is parallel to the y-axis,
            in which case will default to (-1, 0, 0).
        """
        # default behaviour of up direction
        view_direction_along_y_axis = (
            view_direction[0],
            view_direction[2],
        ) == (0, 0)
        up_direction_along_y_axis = (up_direction[0], up_direction[2]) == (
            0,
            0,
        )
        if view_direction_along_y_axis and up_direction_along_y_axis:
            up_direction = (1, 0, 0)  # align up direction along z axis

        # xyz ordering for vispy
        view_vector = np.array(view_direction, dtype=float, copy=True)[::-1]
        # flip z axis for right-handed frame
        view_vector *= [1, 1, -1]
        # normalise vector for rotation matrix
        view_vector /= np.linalg.norm(view_vector)

        # xyz ordering for vispy
        up_vector = np.array(up_direction, dtype=float, copy=True)[::-1]
        # flip z axis for right-handed frame
        up_vector *= [1, 1, -1]
        # ??? why a cross product here?
        up_vector = np.cross(view_vector, up_vector)
        # normalise vector for rotation matrix
        up_vector /= np.linalg.norm(up_vector)

        # explicit check for parallel view direction and up direction
        if np.allclose(np.cross(view_vector, up_vector), 0):
            raise ValueError(
                trans._(
                    'view direction and up direction are parallel',
                    deferred=True,
                )
            )

        x_vector = np.cross(up_vector, view_vector)
        x_vector /= np.linalg.norm(x_vector)

        # construct rotation matrix, convert to euler angles
        rotation_matrix = np.column_stack((up_vector, view_vector, x_vector))
        euler_angles = R.from_matrix(rotation_matrix).as_euler(
            seq='yzx', degrees=True
        )
        self.angles = euler_angles

    def calculate_nd_view_direction(
        self, ndim: int, dims_displayed: tuple[int, ...]
    ) -> np.ndarray | None:
        """Calculate the nD view direction vector of the camera.

        Parameters
        ----------
        ndim : int
            Number of dimensions in which to embed the 3D view vector.
        dims_displayed : Tuple[int]
            Dimensions in which to embed the 3D view vector.

        Returns
        -------
        view_direction_nd : np.ndarray
            nD view direction vector as an (ndim, ) ndarray
        """
        if len(dims_displayed) != 3:
            return None
        view_direction_nd = np.zeros(ndim)
        view_direction_nd[list(dims_displayed)] = self.view_direction
        return view_direction_nd

    def calculate_nd_up_direction(
        self, ndim: int, dims_displayed: tuple[int, ...]
    ) -> np.ndarray | None:
        """Calculate the nD up direction vector of the camera.

        Parameters
        ----------
        ndim : int
            Number of dimensions in which to embed the 3D view vector.
        dims_displayed : Tuple[int]
            Dimensions in which to embed the 3D view vector.

        Returns
        -------
        up_direction_nd : np.ndarray
            nD view direction vector as an (ndim, ) ndarray
        """
        if len(dims_displayed) != 3:
            return None
        up_direction_nd = np.zeros(ndim)
        up_direction_nd[list(dims_displayed)] = self.up_direction
        return up_direction_nd

    @property
    def interactive(self) -> bool:
        warnings.warn(
            '`Camera.interactive` is deprecated since 0.5.0 and will be removed in 0.6.0.',
            category=DeprecationWarning,
        )
        return self.mouse_pan or self.mouse_zoom

    @interactive.setter
    def interactive(self, interactive: bool):
        warnings.warn(
            '`Camera.interactive` is deprecated since 0.5.0 and will be removed in 0.6.0.',
            category=DeprecationWarning,
        )
        self.mouse_pan = interactive
        self.mouse_zoom = interactive
