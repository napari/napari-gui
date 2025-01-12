import sys
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from napari._pydantic_compat import validator
from napari.utils.events import EventedModel, SelectableEventedList
from napari.utils.geometry import intersect_line_with_plane_3d
from napari.utils.translations import trans

if sys.version_info < (3, 10):
    # Once 3.12+ there is a new syntax, type Foo = Bar[...], but
    # we are not there yet.
    from typing import TypeAlias
else:
    from typing import TypeAlias


Point3D: TypeAlias = tuple[float, float, float]


class Plane(EventedModel):
    """Defines a Plane in 3D.

    A Plane is defined by a position, a normal vector and can
    be toggled on or off.

    Attributes
    ----------
    position : 3-tuple
        A 3D position on the plane, defined in sliced data coordinates (currently displayed dims).
    normal : 3-tuple
        A 3D unit vector normal to the plane, defined in sliced data coordinates (currently displayed dims).
    enabled : bool
        Whether the plane is considered enabled.
    """

    normal: Point3D = (1, 0, 0)
    position: Point3D = (0, 0, 0)

    @validator('normal', allow_reuse=True)
    def _normalise_vector(cls, v: npt.NDArray) -> Point3D:
        return cast(Point3D, tuple(v / np.linalg.norm(v)))

    @validator('normal', 'position', pre=True, allow_reuse=True)
    def _ensure_tuple(cls, v: Any) -> Point3D:
        return cast(Point3D, tuple(v))

    def shift_along_normal_vector(self, distance: float) -> None:
        """Shift the plane along its normal vector by a given distance."""
        assert len(self.position) == len(self.normal) == 3
        self.position = cast(
            Point3D,
            tuple(
                p + (distance * n)
                for p, n in zip(self.position, self.normal, strict=False)
            ),
        )

    def intersect_with_line(
        self, line_position: np.ndarray, line_direction: np.ndarray
    ) -> np.ndarray:
        """Calculate a 3D line-plane intersection."""
        return intersect_line_with_plane_3d(
            line_position, line_direction, self.position, self.normal
        )

    @classmethod
    def from_points(
        cls,
        a: npt.NDArray,
        b: npt.NDArray,
        c: npt.NDArray,
        enabled: bool = True,
    ) -> 'Plane':
        """Derive a Plane from three points.

        Parameters
        ----------
        a : ArrayLike
            (3,) array containing coordinates of a point
        b : ArrayLike
            (3,) array containing coordinates of a point
        c : ArrayLike
            (3,) array containing coordinates of a point

        Returns
        -------
        plane : Plane
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        abc = np.vstack((a, b, c))
        ab = b - a
        ac = c - a

        plane_normal = np.cross(ab, ac)
        plane_position = np.mean(abc, axis=0)
        return cls(
            position=plane_position, normal=plane_normal, enabled=enabled
        )

    def as_array(self) -> npt.NDArray:
        """Return a (2, 3) array representing the plane.

        [0, :] : plane position
        [1, :] : plane normal
        """
        return np.stack([self.position, self.normal])

    @classmethod
    def from_array(cls, array: npt.NDArray, enabled: bool = True) -> 'Plane':
        """Construct a plane from a (2, 3) array.

        [0, :] : plane position
        [1, :] : plane normal
        """
        return cls(position=array[0], normal=array[1], enabled=enabled)

    def __hash__(self) -> int:
        return id(self)


class SlicingPlane(Plane):
    """Defines a draggable plane in 3D with a defined thickness.

    A slicing plane is defined by a position, a normal vector and a thickness
    value.

    Attributes
    ----------
    position : 3-tuple
        A 3D position on the plane, defined in sliced data coordinates
        (currently displayed dims).
    normal : 3-tuple
        A 3D unit vector normal to the plane, defined in sliced data coordinates
        (currently displayed dims).
    thickness : float
        Thickness of the slice.
    """

    thickness: float = 0.0


class ClippingPlane(Plane):
    """Defines a clipping plane in 3D.

    A clipping plane is defined by a position, a normal vector and can
    be toggled on or off.

    Attributes
    ----------
    position : 3-tuple
        A 3D position on the plane, defined in sliced data coordinates (currently displayed dims).
    normal : 3-tuple
        A 3D unit vector normal to the plane, defined in sliced data coordinates (currently displayed dims).
    enabled : bool
        Whether the plane is considered enabled.
    """

    enabled: bool = True


class ClippingPlaneList(SelectableEventedList):
    """A list of planes with some utility methods."""

    def as_array(self) -> npt.NDArray:
        """Return a (N, 2, 3) array of clipping planes.

        [i, 0, :] : ith plane position
        [i, 1, :] : ith plane normal
        """
        arrays = []
        for plane in self:
            if plane.enabled:
                arrays.append(plane.as_array())
        if not arrays:
            return np.empty((0, 2, 3))
        return np.stack(arrays)

    @classmethod
    def from_array(
        cls, array: npt.NDArray, enabled: bool = True
    ) -> 'ClippingPlaneList':
        """Construct the PlaneList from an (N, 2, 3) array.

        [i, 0, :] : ith plane position
        [i, 1, :] : ith plane normal
        """
        if array.ndim != 3 or array.shape[1:] != (2, 3):
            raise ValueError(
                trans._(
                    'Planes can only be constructed from arrays of shape (N, 2, 3), not {shape}',
                    deferred=True,
                    shape=array.shape,
                )
            )
        planes = [
            ClippingPlane.from_array(sub_arr, enabled=enabled)
            for sub_arr in array
        ]
        return cls(planes)

    @classmethod
    def from_bounding_box(
        cls, center: Point3D, dimensions: Point3D, enabled: bool = True
    ) -> 'ClippingPlaneList':
        """
        generate 6 planes positioned to form a bounding box, with normals towards the center

        Parameters
        ----------
        center : ArrayLike
            (3,) array, coordinates of the center of the box
        dimensions : ArrayLike
            (3,) array, dimensions of the box

        Returns
        -------
        list : ClippingPlaneList
        """
        planes = []
        for axis in range(3):
            for direction in (-1, 1):
                shift = (dimensions[axis] / 2) * direction
                position = np.array(center)
                position[axis] += shift

                normal = np.zeros(3)
                normal[axis] = -direction

                planes.append(
                    ClippingPlane(
                        position=position, normal=normal, enabled=enabled
                    )
                )
        return cls(planes)

    def add_plane(self, **kwargs: Any) -> None:
        """Add a clipping plane to the list."""
        self.append(ClippingPlane(**kwargs))
