from typing import Sequence

import numpy as np
import toolz as tz

from ..utils.list import ListModel
from .utils.transform_utils import (
    compose_linear_matrix,
    decompose_linear_matrix,
    embed_in_identity_matrix,
)


class Transform:
    """Base transform class.

    Defaults to the identity transform.

    Parameters
    ----------
    func : callable, Coords -> Coords
        A function converting an NxD array of coordinates to NxD'.
    name : string
        A string name for the transform.
    """

    def __init__(self, func=tz.identity, inverse=None, name=None):
        self.func = func
        self._inverse_func = inverse
        self.name = name

        if func is tz.identity:
            self._inverse_func = tz.identity

    def __call__(self, coords):
        """Transform input coordinates to output."""
        return self.func(coords)

    @property
    def inverse(self) -> 'Transform':
        if self._inverse_func is not None:
            return Transform(self._inverse_func, self.func)
        else:
            raise ValueError('Inverse function was not provided.')

    def compose(self, transform: 'Transform') -> 'Transform':
        """Return the composite of this transform and the provided one."""
        raise ValueError('Transform composition rule not provided')

    def set_slice(self, axes: Sequence[int]) -> 'Transform':
        """Return a transform subset to the visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Axes to subset the current transform with.

        Returns
        -------
        Transform
            Resulting transform.
        """
        raise NotImplementedError('Cannot subset arbitrary transforms.')

    def expand_dims(self, axes: Sequence[int]) -> 'Transform':
        """Return a transform with added axes for non-visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Location of axes to expand the current transform with. Passing a
            list allows expansion to occur at specific locations and for
            expand_dims to be like an inverse to the set_slice method.

        Returns
        -------
        Transform
            Resulting transform.
        """
        raise NotImplementedError('Cannot subset arbitrary transforms.')


class TransformChain(ListModel, Transform):
    def __init__(self, transforms=[]):
        super().__init__(
            basetype=Transform,
            iterable=transforms,
            lookup={str: lambda q, e: q == e.name},
        )

    def __call__(self, coords):
        return tz.pipe(coords, *self)

    def __newlike__(self, iterable):
        return TransformChain(iterable)

    @property
    def inverse(self) -> 'TransformChain':
        """Return the inverse transform chain."""
        return TransformChain([tf.inverse for tf in self[::-1]])

    @property
    def simplified(self) -> 'Transform':
        """Return the composite of the transforms inside the transform chain."""
        if len(self) == 0:
            return None
        if len(self) == 1:
            return self[0]
        else:
            return tz.pipe(self[0], *[tf.compose for tf in self[1:]])

    def set_slice(self, axes: Sequence[int]) -> 'TransformChain':
        """Return a transform chain subset to the visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Axes to subset the current transform chain with.

        Returns
        -------
        TransformChain
            Resulting transform chain.
        """
        return TransformChain([tf.set_slice(axes) for tf in self])

    def expand_dims(self, axes: Sequence[int]) -> 'Transform':
        """Return a transform chain with added axes for non-visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Location of axes to expand the current transform with. Passing a
            list allows expansion to occur at specific locations and for
            expand_dims to be like an inverse to the set_slice method.

        Returns
        -------
        TransformChain
            Resulting transform chain.
        """
        return TransformChain([tf.expand_dims(axes) for tf in self])


class ScaleTranslate(Transform):
    """n-dimensional scale and translation (shift) class.

    Scaling is always applied before translation.

    Parameters
    ----------
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    translate : 1-D array
        A 1-D array of factors to shift each axis by. Translation is broadcast
        to 0 in leading dimensions, so that, for example, a translation of
        [4, 18, 34] in 3D can be used as a translation of [0, 4, 18, 34] in 4D
        without modification. An empty translation vector implies no
        translation.
    name : string
        A string name for the transform.
    """

    def __init__(self, scale=(1.0,), translate=(0.0,), *, name=None):
        super().__init__(name=name)
        self.scale = np.array(scale)
        self.translate = np.array(translate)

    def __call__(self, coords):
        coords = np.atleast_2d(coords)
        scale = np.concatenate(
            ([1.0] * (coords.shape[1] - len(self.scale)), self.scale)
        )
        translate = np.concatenate(
            ([0.0] * (coords.shape[1] - len(self.translate)), self.translate)
        )
        return np.atleast_1d(np.squeeze(scale * coords + translate))

    @property
    def inverse(self) -> 'ScaleTranslate':
        """Return the inverse transform."""
        return ScaleTranslate(1 / self.scale, -1 / self.scale * self.translate)

    def compose(self, transform: 'ScaleTranslate') -> 'ScaleTranslate':
        """Return the composite of this transform and the provided one."""
        scale = self.scale * transform.scale
        translate = self.translate + self.scale * transform.translate
        return ScaleTranslate(scale, translate)

    def set_slice(self, axes: Sequence[int]) -> 'ScaleTranslate':
        """Return a transform subset to the visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Axes to subset the current transform with.

        Returns
        -------
        Transform
            Resulting transform.
        """
        return ScaleTranslate(
            self.scale[axes], self.translate[axes], name=self.name
        )

    def expand_dims(self, axes: Sequence[int]) -> 'ScaleTranslate':
        """Return a transform with added axes for non-visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Location of axes to expand the current transform with. Passing a
            list allows expansion to occur at specific locations and for
            expand_dims to be like an inverse to the set_slice method.

        Returns
        -------
        Transform
            Resulting transform.
        """
        n = len(axes) + len(self.scale)
        not_axes = [i for i in range(n) if i not in axes]
        scale = np.ones(n)
        scale[not_axes] = self.scale
        translate = np.zeros(n)
        translate[not_axes] = self.translate
        return ScaleTranslate(scale, translate, name=self.name)


class Affine(Transform):
    """n-dimensional affine transformation class.

    The affine transform is represented as a n+1 dimensionsal linear_matrix

    Parameters
    ----------
    rotation : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation linear_matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation linear_matrix, rolling a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angle
        conversion are done either using degrees or radians depending on the
        degrees boolean parameter.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : 1-D array
        A vector of shear values for an upper triangular n-D shear linear_matrix.
    translate : 1-D array
        A 1-D array of factors to shift each axis by. Translation is broadcast
        to 0 in leading dimensions, so that, for example, a translation of
        [4, 18, 34] in 3D can be used as a translation of [0, 4, 18, 34] in 4D
        without modification. An empty translation vector implies no
        translation.
    linear_matrix : n-D array, optional
        (N, N) matrix with linear transform. If provided then scale, rotation,
        and shear values are ignored.
    affine_matrix : n-D array, optional
        (N+1, N+1) matrix where first (N, N) entries correspond to a linear
        transform and the final column is a lenght N translation vector and
        a 1. If provided then linear_matrix, scale, rotation, and shear values
        are ignored.
    degrees : bool
        Boolean if rotation angles are provided in degrees
    name : string
        A string name for the transform.
    """

    def __init__(
        self,
        scale=(1.0,),
        translate=(0.0,),
        *,
        rotation=None,
        shear=None,
        linear_matrix=None,
        affine_matrix=None,
        degrees=True,
        name=None,
    ):
        super().__init__(name=name)

        if affine_matrix is not None:
            linear_matrix = affine_matrix[:-1, :-1]
            translate = affine_matrix[:-1, -1]
        elif linear_matrix is not None:
            linear_matrix = np.array(linear_matrix)
        else:
            if rotation is None:
                rotation = np.eye(len(scale))
            if shear is None:
                shear = np.eye(len(scale))
            linear_matrix = compose_linear_matrix(
                rotation, scale, shear, degrees=degrees
            )

        ndim = max(linear_matrix.shape[0], len(translate))
        self.linear_matrix = embed_in_identity_matrix(linear_matrix, ndim)
        self.translate = np.array(
            [0] * (ndim - len(translate)) + list(translate)
        )

    def __call__(self, coords):
        coords = np.atleast_2d(coords)
        if coords.shape[1] != self.linear_matrix.shape[0]:
            linear_matrix = np.eye(coords.shape[1])
            linear_matrix[
                -self.linear_matrix.shape[0] :, -self.linear_matrix.shape[1] :
            ] = self.linear_matrix
        else:
            linear_matrix = self.linear_matrix
        translate = np.concatenate(
            ([0.0] * (coords.shape[1] - len(self.translate)), self.translate)
        )
        return np.atleast_1d(np.squeeze(coords @ linear_matrix.T + translate))

    @property
    def ndim(self) -> int:
        """Dimensionality of the transform."""
        return self.linear_matrix.shape[0]

    @property
    def scale(self) -> np.array:
        """Return the scale of the transform."""
        return decompose_linear_matrix(self.linear_matrix)[1]

    @scale.setter
    def scale(self, scale):
        """Set the scale of the transform."""
        rotation, _, shear = decompose_linear_matrix(self.linear_matrix)
        self.linear_matrix = compose_linear_matrix(rotation, scale, shear)

    @property
    def rotation(self) -> np.array:
        """Return the rotation of the transform."""
        return decompose_linear_matrix(self.linear_matrix)[0]

    @rotation.setter
    def rotation(self, rotation):
        """Set the rotation of the transform."""
        _, scale, shear = decompose_linear_matrix(self.linear_matrix)
        self.linear_matrix = compose_linear_matrix(rotation, scale, shear)

    @property
    def shear(self) -> np.array:
        """Return the shear of the transform."""
        return decompose_linear_matrix(self.linear_matrix)[2]

    @shear.setter
    def shear(self, shear):
        """Set the shear of the transform."""
        rotation, scale, _ = decompose_linear_matrix(self.linear_matrix)
        self.linear_matrix = compose_linear_matrix(rotation, scale, shear)

    @property
    def affine_matrix(self) -> np.array:
        """Return the affine matrix for the transform."""
        matrix = np.eye(self.ndim + 1, self.ndim + 1)
        matrix[:-1, :-1] = self.linear_matrix
        matrix[:-1, -1] = self.translate
        return matrix

    @affine_matrix.setter
    def affine_matrix(self, affine_matrix):
        """Set the affine matrix for the transform."""
        self.linear_matrix = affine_matrix[:-1, :-1]
        self.translate = affine_matrix[:-1, -1]

    @property
    def inverse(self) -> 'Affine':
        """Return the inverse transform."""
        linear_matrix = np.linalg.inv(self.linear_matrix)
        translate = -self.translate @ linear_matrix.T
        return Affine(linear_matrix=linear_matrix, translate=translate)

    def compose(self, transform: 'Affine') -> 'Affine':
        """Return the composite of this transform and the provided one."""
        linear_matrix = self.linear_matrix @ transform.linear_matrix
        translate = self.translate + transform.translate @ self.linear_matrix.T
        return Affine(linear_matrix=linear_matrix, translate=translate)

    def set_slice(self, axes: Sequence[int]) -> 'Affine':
        """Return a transform subset to the visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Axes to subset the current transform with.

        Returns
        -------
        Transform
            Resulting transform.
        """
        return Affine(
            linear_matrix=self.linear_matrix[np.ix_(axes, axes)],
            translate=self.translate[axes],
            name=self.name,
        )

    def expand_dims(self, axes: Sequence[int]) -> 'Affine':
        """Return a transform with added axes for non-visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Location of axes to expand the current transform with. Passing a
            list allows expansion to occur at specific locations and for
            expand_dims to be like an inverse to the set_slice method.

        Returns
        -------
        Transform
            Resulting transform.
        """
        n = len(axes) + len(self.scale)
        not_axes = [i for i in range(n) if i not in axes]
        linear_matrix = np.eye(n)
        linear_matrix[np.ix_(not_axes, not_axes)] = self.linear_matrix
        translate = np.zeros(n)
        translate[not_axes] = self.translate
        return Affine(
            linear_matrix=linear_matrix, translate=translate, name=self.name
        )
