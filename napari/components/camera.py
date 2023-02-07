from typing import Optional, Tuple, Union

import numpy as np
from pydantic import validator
from scipy.spatial.transform import Rotation as R

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
        If the camera interactivity is enabled or not.
    """

    # fields
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    zoom: float = 1.0
    angles: Tuple[float, float, float] = (0.0, 0.0, 90.0)
    perspective: float = 0
    interactive: bool = True

    # validators
    @validator('center', 'angles', pre=True)
    def _ensure_3_tuple(v):
        return ensure_n_tuple(v, n=3)

    @property
    def view_direction(self) -> Tuple[float, float, float]:
        """3D view direction vector of the camera.

        View direction is calculated from the Euler angles and returned as a
        3-tuple. This direction is in 3D scene coordinates, the world coordinate
        system for three currently displayed dimensions.
        """
        ang = np.deg2rad(self.angles)
        view_direction = (
            np.sin(ang[2]) * np.cos(ang[1]),
            np.cos(ang[2]) * np.cos(ang[1]),
            -np.sin(ang[1]),
        )
        return view_direction

    @property
    def up_direction(self) -> Tuple[float, float, float]:
        """3D direction vector pointing up on the canvas.

        Up direction is calculated from the Euler angles and returned as a
        3-tuple. This direction is in 3D scene coordinates, the world coordinate
        system for three currently displayed dimensions.
        """
        rotation_matrix = R.from_euler(
            seq='yzx', angles=self.angles, degrees=True
        ).as_matrix()
        return tuple(rotation_matrix[:, 2][::-1])

    def set_view_direction(
        self,
        view_direction: Tuple[float, float, float],
        up_direction: Tuple[float, float, float] = (0, -1, 0),
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
            up_direction = (-1, 0, 0)  # align up direction along z axis

        # xyz ordering for vispy, normalise vectors for rotation matrix
        view_direction = np.asarray(view_direction, dtype=float)[::-1]
        view_direction /= np.linalg.norm(view_direction)

        up_direction = np.asarray(up_direction, dtype=float)[::-1]
        up_direction = np.cross(view_direction, up_direction)
        up_direction /= np.linalg.norm(up_direction)

        # explicit check for parallel view direction and up direction
        if np.allclose(np.cross(view_direction, up_direction), 0):
            raise ValueError(
                trans._(
                    "view direction and up direction are parallel",
                    deferred=True,
                )
            )

        x_direction = np.cross(up_direction, view_direction)
        x_direction /= np.linalg.norm(x_direction)

        # construct rotation matrix, convert to euler angles
        rotation_matrix = np.column_stack(
            (up_direction, view_direction, x_direction)
        )
        euler_angles = R.from_matrix(rotation_matrix).as_euler(
            seq='yzx', degrees=True
        )
        self.angles = euler_angles

    def look_at(
        self, point: Union[Tuple[float, float], Tuple[float, float, float]]
    ):
        """Point the camera at a point in the 2D/3D scene.

        In 2D, this method simply sets the camera center position. In 3D,
        this keeps the camera center the same but updates the view direction,
        ensuring that the point supplied as an argument is centered.
        """
        if len(point) == 2:
            self.center = point
        elif len(point) == 3:
            view_direction = np.asarray(point) - np.asarray(self.center)
            self.set_view_direction(
                view_direction, up_direction=self.up_direction
            )
        else:
            raise ValueError("Point must be 2D or 3D.")

    def calculate_nd_view_direction(
        self, ndim: int, dims_displayed: Tuple[int]
    ) -> np.ndarray:
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
        self, ndim: int, dims_displayed: Tuple[int]
    ) -> Optional[np.ndarray]:
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
