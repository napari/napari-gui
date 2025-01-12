from collections.abc import Iterable, Sequence
from typing import Generic, TypeVar, overload

import numpy as np
import numpy.typing as npt
import pint
import toolz as tz
from psygnal import Signal

from napari.utils.events import EventedList
from napari.utils.transforms._units import get_units_from_name
from napari.utils.transforms.transform_utils import (
    compose_linear_matrix,
    decompose_linear_matrix,
    embed_in_identity_matrix,
    infer_ndim,
    is_diagonal,
    is_matrix_triangular,
    is_matrix_upper_triangular,
    rotate_to_matrix,
    scale_to_vector,
    shear_to_matrix,
    translate_to_vector,
)
from napari.utils.translations import trans


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

    changed = Signal()

    def __init__(self, func=tz.identity, inverse=None, name=None) -> None:
        self.func = func
        self._inverse_func = inverse
        self.name = name
        self._cache_dict = {}

        if func is tz.identity:
            self._inverse_func = tz.identity

    def __call__(self, coords):
        """Transform input coordinates to output."""
        return self.func(coords)

    @property
    def inverse(self) -> 'Transform':
        if self._inverse_func is None:
            raise ValueError(
                trans._('Inverse function was not provided.', deferred=True)
            )
        if 'inverse' not in self._cache_dict:
            self._cache_dict['inverse'] = Transform(
                self._inverse_func, self.func
            )
        return self._cache_dict['inverse']

    def compose(self, transform: 'Transform') -> 'Transform':
        """Return the composite of this transform and the provided one."""
        return TransformChain([self, transform])

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
        raise NotImplementedError(
            trans._('Cannot subset arbitrary transforms.', deferred=True)
        )

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
        raise NotImplementedError(
            trans._('Cannot subset arbitrary transforms.', deferred=True)
        )

    @property
    def _is_diagonal(self):
        """Indicate when a transform does not mix or permute dimensions.

        Can be overridden in subclasses to enable performance optimizations
        that are specific to this case.
        """
        return False

    def _clean_cache(self):
        self._cache_dict.clear()
        self.changed.emit()


_T = TypeVar('_T', bound=Transform)


class TransformChain(EventedList[_T], Transform, Generic[_T]):
    def __init__(self, transforms: Iterable[Transform] | None = None) -> None:
        if transforms is None:
            transforms = []
        super().__init__(
            data=transforms,
            basetype=Transform,
            lookup={str: lambda x: x.name},
        )
        # The above super().__init__() will not call Transform.__init__().
        # For that to work every __init__() called using super() needs to
        # in turn call super().__init__(). So we call it explicitly here.
        Transform.__init__(self)
        for tr in self:
            if hasattr(tr, 'changed'):
                tr.changed.connect(self._clean_cache)

    def __call__(self, coords):
        return tz.pipe(coords, *self)

    def __newlike__(self, iterable):
        return TransformChain(iterable)

    @overload
    def __getitem__(self, key: int) -> _T: ...

    @overload
    def __getitem__(self, key: str) -> _T: ...

    @overload
    def __getitem__(self, key: slice) -> 'TransformChain[_T]': ...

    def __getitem__(self, key):
        if f'getitem_{key}' not in self._cache_dict:
            self._cache_dict[f'getitem_{key}'] = super().__getitem__(key)
        return self._cache_dict[f'getitem_{key}']

    def __setitem__(self, key, value):
        if key in self and hasattr(self[key], 'changed'):
            self[key].changed.disconnect(self._clean_cache)
        super().__setitem__(key, value)
        if hasattr(value, 'changed'):
            value.changed.connect(self._clean_cache)
        self._clean_cache()

    def __delitem__(self, key):
        val = self[key]
        if hasattr(val, 'changed'):
            val.changed.disconnect(self._clean_cache)
        super().__delitem__(key)
        self._clean_cache()

    @property
    def inverse(self) -> 'TransformChain':
        """Return the inverse transform chain."""
        if 'inverse' not in self._cache_dict:
            self._cache_dict['inverse'] = TransformChain(
                [tf.inverse for tf in self[::-1]]
            )
        return self._cache_dict['inverse']

    @property
    def _is_diagonal(self):
        if all(getattr(tf, '_is_diagonal', False) for tf in self):
            return True
        return getattr(self.simplified, '_is_diagonal', False)

    @property
    def simplified(self) -> _T:
        """
        Return the composite of the transforms inside the transform chain.

        Raises
        ------
        ValueError
            If the transform chain is empty.
        """
        if len(self) == 0:
            raise ValueError(
                trans._('Cannot simplify an empty transform chain.')
            )

        if len(self) == 1:
            return self[0]

        if 'simplified' not in self._cache_dict:
            self._cache_dict['simplified'] = tz.pipe(
                self[0], *[tf.compose for tf in self[1:]]
            )
        return self._cache_dict['simplified']

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

    def expand_dims(self, axes: Sequence[int]) -> 'TransformChain':
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

    def __init__(self, scale=(1.0,), translate=(0.0,), *, name=None) -> None:
        super().__init__(name=name)

        if len(scale) > len(translate):
            translate = [0] * (len(scale) - len(translate)) + list(translate)

        if len(translate) > len(scale):
            scale = [1] * (len(translate) - len(scale)) + list(scale)

        self.scale = np.array(scale)
        self.translate = np.array(translate)

    def __call__(self, coords):
        coords = np.asarray(coords)
        append_first_axis = coords.ndim == 1
        if append_first_axis:
            coords = coords[np.newaxis, :]

        coords_ndim = coords.shape[1]
        if coords_ndim == len(self.scale):
            scale = self.scale
            translate = self.translate
        else:
            scale = np.concatenate(
                ([1.0] * (coords_ndim - len(self.scale)), self.scale)
            )
            translate = np.concatenate(
                ([0.0] * (coords_ndim - len(self.translate)), self.translate)
            )
        out = scale * coords
        out += translate
        if append_first_axis:
            out = out[0]
        return out

    @property
    def inverse(self) -> 'ScaleTranslate':
        """Return the inverse transform."""
        return ScaleTranslate(1 / self.scale, -1 / self.scale * self.translate)

    def compose(self, transform: 'Transform') -> 'Transform':
        """Return the composite of this transform and the provided one."""
        if not isinstance(transform, ScaleTranslate):
            super().compose(transform)
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

    @property
    def _is_diagonal(self):
        """Indicate that this transform does not mix or permute dimensions."""
        return True


class Affine(Transform):
    """n-dimensional affine transformation class.

    The affine transform can be represented as a n+1 dimensional
    transformation matrix in homogeneous coordinates [1]_, an n
    dimensional matrix and a length n translation vector, or be
    composed and decomposed from scale, rotate, and shear
    transformations defined in the following order:

    rotate * shear * scale + translate

    The affine_matrix representation can be used for easy compatibility
    with other libraries that can generate affine transformations.

    Parameters
    ----------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    translate : 1-D array
        A 1-D array of factors to shift each axis by. Translation is broadcast
        to 0 in leading dimensions, so that, for example, a translation of
        [4, 18, 34] in 3D can be used as a translation of [0, 4, 18, 34] in 4D
        without modification. An empty translation vector implies no
        translation.
    linear_matrix : n-D array, optional
        (N, N) matrix with linear transform. If provided then scale, rotate,
        and shear values are ignored.
    affine_matrix : n-D array, optional
        (N+1, N+1) affine transformation matrix in homogeneous coordinates [1]_.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        AffineTransform object. If provided then translate, scale, rotate, and
        shear values are ignored.
    ndim : int
        The dimensionality of the transform. If None, this is inferred from the
        other parameters.
    name : string
        A string name for the transform.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates.
    """

    def __init__(
        self,
        scale=(1.0, 1.0),
        translate=(
            0.0,
            0.0,
        ),
        *,
        affine_matrix=None,
        axis_labels: Sequence[str] | None = None,
        linear_matrix=None,
        name=None,
        ndim=None,
        rotate=None,
        shear=None,
        units: Sequence[str | pint.Unit] | None = None,
    ) -> None:
        super().__init__(name=name)
        self._upper_triangular = True
        if ndim is None:
            ndim = infer_ndim(
                scale=scale, translate=translate, rotate=rotate, shear=shear
            )

        if affine_matrix is not None:
            linear_matrix = affine_matrix[:-1, :-1]
            translate = affine_matrix[:-1, -1]
        elif linear_matrix is not None:
            linear_matrix = np.array(linear_matrix)
        else:
            if rotate is None:
                rotate = np.eye(ndim)
            if shear is None:
                shear = np.eye(ndim)
            else:
                if np.array(shear).ndim == 2:
                    if is_matrix_triangular(shear):
                        self._upper_triangular = is_matrix_upper_triangular(
                            shear
                        )
                    else:
                        raise ValueError(
                            trans._(
                                'Only upper triangular or lower triangular matrices are accepted for shear, got {shear}. For other matrices, set the affine_matrix or linear_matrix directly.',
                                deferred=True,
                                shear=shear,
                            )
                        )
            linear_matrix = compose_linear_matrix(rotate, scale, shear)

        ndim = max(ndim, linear_matrix.shape[0])
        self._linear_matrix = embed_in_identity_matrix(linear_matrix, ndim)
        self._translate = translate_to_vector(translate, ndim=ndim)
        self._axis_labels = tuple(f'axis {i}' for i in range(-ndim, 0))
        self._units = (pint.get_application_registry().pixel,) * ndim

        self.axis_labels = axis_labels
        self.units = units

    def __call__(self, coords):
        coords = np.asarray(coords)
        append_first_axis = coords.ndim == 1
        if append_first_axis:
            coords = coords[np.newaxis, :]
        coords_ndim = coords.shape[1]
        padded_linear_matrix = embed_in_identity_matrix(
            self._linear_matrix, coords_ndim
        )
        translate = translate_to_vector(self._translate, ndim=coords_ndim)
        out = coords @ padded_linear_matrix.T
        out += translate
        if append_first_axis:
            out = out[0]
        return out

    @property
    def ndim(self) -> int:
        """Dimensionality of the transform."""
        return self._linear_matrix.shape[0]

    @property
    def axis_labels(self) -> tuple[str, ...]:
        """tuple of axis labels for the layer."""
        return self._axis_labels

    @axis_labels.setter
    def axis_labels(self, axis_labels: Sequence[str] | None) -> None:
        if axis_labels is None:
            axis_labels = tuple(f'axis {i}' for i in range(-self.ndim, 0))
        if len(axis_labels) != self.ndim:
            raise ValueError(
                f'{axis_labels=} must have length ndim={self.ndim}.'
            )
        axis_labels = tuple(axis_labels)
        self._axis_labels = axis_labels

    @property
    def units(self) -> tuple[pint.Unit, ...]:
        """List of units for the layer."""
        return self._units

    @units.setter
    def units(self, units: Sequence[pint.Unit] | None) -> None:
        units = get_units_from_name(units)
        if units is None:
            units = (pint.get_application_registry().pixel,) * self.ndim
        if isinstance(units, pint.Unit):
            units = (units,) * self.ndim
        if len(units) != self.ndim:
            raise ValueError(f'{units=} must have length ndim={self.ndim}.')
        self._units = units

    @property
    def scale(self) -> npt.NDArray:
        """Return the scale of the transform."""
        if self._is_diagonal:
            return np.diag(self._linear_matrix)
        self._setup_decompose_linear_matrix_cache()
        return self._cache_dict['decompose_linear_matrix'][1]

    @scale.setter
    def scale(self, scale):
        """Set the scale of the transform."""
        if self._is_diagonal:
            scale = scale_to_vector(scale, ndim=self.ndim)
            for i in range(len(scale)):
                self._linear_matrix[i, i] = scale[i]
        else:
            self._linear_matrix = compose_linear_matrix(
                self.rotate, scale, self._shear_cache
            )
        self._clean_cache()

    @property
    def physical_scale(self) -> tuple[pint.Quantity, ...]:
        """Return the scale of the transform, with units."""
        return tuple(np.multiply(self.scale, self.units))

    @property
    def translate(self) -> npt.NDArray:
        """Return the translation of the transform."""
        return self._translate

    @translate.setter
    def translate(self, translate):
        """Set the translation of the transform."""
        self._translate = translate_to_vector(translate, ndim=self.ndim)
        self._clean_cache()

    def _setup_decompose_linear_matrix_cache(self):
        if 'decompose_linear_matrix' in self._cache_dict:
            return
        self._cache_dict['decompose_linear_matrix'] = decompose_linear_matrix(
            self.linear_matrix, upper_triangular=self._upper_triangular
        )

    @property
    def rotate(self) -> npt.NDArray:
        """Return the rotation of the transform."""
        self._setup_decompose_linear_matrix_cache()
        return self._cache_dict['decompose_linear_matrix'][0]

    @rotate.setter
    def rotate(self, rotate):
        """Set the rotation of the transform."""
        self._linear_matrix = compose_linear_matrix(
            rotate, self.scale, self._shear_cache
        )
        self._clean_cache()

    @property
    def shear(self) -> npt.NDArray:
        """Return the shear of the transform."""
        if self._is_diagonal:
            return np.zeros((self.ndim,))
        self._setup_decompose_linear_matrix_cache()
        return self._cache_dict['decompose_linear_matrix'][2]

    @shear.setter
    def shear(self, shear):
        """Set the shear of the transform."""
        shear = np.asarray(shear)
        if shear.ndim == 2:
            if is_matrix_triangular(shear):
                self._upper_triangular = is_matrix_upper_triangular(shear)
            else:
                raise ValueError(
                    trans._(
                        'Only upper triangular or lower triangular matrices are accepted for shear, got {shear}. For other matrices, set the affine_matrix or linear_matrix directly.',
                        deferred=True,
                        shear=shear,
                    )
                )
        else:
            self._upper_triangular = True
        self._linear_matrix = compose_linear_matrix(
            self.rotate, self.scale, shear
        )
        self._clean_cache()

    @property
    def _shear_cache(self):
        self._setup_decompose_linear_matrix_cache()
        return self._cache_dict['decompose_linear_matrix'][2]

    @property
    def linear_matrix(self) -> npt.NDArray:
        """Return the linear matrix of the transform."""
        return self._linear_matrix

    @linear_matrix.setter
    def linear_matrix(self, linear_matrix):
        """Set the linear matrix of the transform."""
        self._linear_matrix = embed_in_identity_matrix(
            linear_matrix, ndim=self.ndim
        )
        self._clean_cache()

    @property
    def affine_matrix(self) -> npt.NDArray:
        """Return the affine matrix for the transform."""
        matrix = np.eye(self.ndim + 1, self.ndim + 1)
        matrix[:-1, :-1] = self._linear_matrix
        matrix[:-1, -1] = self._translate
        return matrix

    @affine_matrix.setter
    def affine_matrix(self, affine_matrix):
        """Set the affine matrix for the transform."""
        self._linear_matrix = affine_matrix[:-1, :-1]
        self._translate = affine_matrix[:-1, -1]
        self._clean_cache()

    def __array__(self, *args, **kwargs):
        """NumPy __array__ protocol to get the affine transform matrix."""
        return self.affine_matrix

    @property
    def inverse(self) -> 'Affine':
        """Return the inverse transform."""
        if 'inverse' not in self._cache_dict:
            self._cache_dict['inverse'] = Affine(
                affine_matrix=np.linalg.inv(self.affine_matrix)
            )
        return self._cache_dict['inverse']

    @overload
    def compose(self, transform: 'Affine') -> 'Affine': ...

    @overload
    def compose(self, transform: 'Transform') -> 'Transform': ...

    def compose(self, transform):
        """Return the composite of this transform and the provided one."""
        if not isinstance(transform, Affine):
            return super().compose(transform)
        affine_matrix = self.affine_matrix @ transform.affine_matrix
        return Affine(affine_matrix=affine_matrix)

    def set_slice(self, axes: Sequence[int]) -> 'Affine':
        """Return a transform subset to the visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Axes to subset the current transform with.

        Returns
        -------
        Affine
            Resulting transform.
        """
        axes = list(axes)
        if self._is_diagonal:
            linear_matrix = np.diag(self.scale[axes])
        else:
            linear_matrix = self.linear_matrix[np.ix_(axes, axes)]
        units = [self.units[i] for i in axes]
        axes_labels = [self.axis_labels[i] for i in axes]
        return Affine(
            linear_matrix=linear_matrix,
            translate=self.translate[axes],
            ndim=len(axes),
            name=self.name,
            units=units,
            axis_labels=axes_labels,
        )

    def replace_slice(
        self, axes: Sequence[int], transform: 'Affine'
    ) -> 'Affine':
        """Returns a transform where the transform at the indicated n dimensions is replaced with another n-dimensional transform

        Parameters
        ----------
        axes : Sequence[int]
            Axes where the transform will be replaced
        transform : Affine
            The transform that will be inserted. Must have as many dimension as len(axes)

        Returns
        -------
        Affine
            Resulting transform.
        """

        if len(axes) != transform.ndim:
            raise ValueError(
                trans._(
                    'Dimensionality of provided axes list and transform differ.',
                    deferred=True,
                )
            )

        linear_matrix = np.copy(self.linear_matrix)
        linear_matrix[np.ix_(axes, axes)] = transform.linear_matrix
        translate = np.copy(self.translate)
        translate[axes] = transform.translate
        return Affine(
            linear_matrix=linear_matrix,
            translate=translate,
            ndim=len(axes),
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
            linear_matrix=linear_matrix,
            translate=translate,
            ndim=n,
            name=self.name,
        )

    @property
    def _is_diagonal(self):
        """Determine whether linear_matrix is diagonal up to some tolerance.

        Since only `self.linear_matrix` is checked, affines with a translation
        component can still be considered diagonal.
        """
        if '_is_diagonal' not in self._cache_dict:
            self._cache_dict['_is_diagonal'] = is_diagonal(
                self.linear_matrix, tol=1e-8
            )
        return self._cache_dict['_is_diagonal']


class CompositeAffine(Affine):
    """n-dimensional affine transformation composed from more basic components.

    Composition is in the following order

    rotate * shear * scale + translate

    Parameters
    ----------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    translate : 1-D array
        A 1-D array of factors to shift each axis by. Translation is broadcast
        to 0 in leading dimensions, so that, for example, a translation of
        [4, 18, 34] in 3D can be used as a translation of [0, 4, 18, 34] in 4D
        without modification. An empty translation vector implies no
        translation.
    ndim : int
        The dimensionality of the transform. If None, this is inferred from the
        other parameters.
    name : string
        A string name for the transform.
    """

    def __init__(
        self,
        scale=(1, 1),
        translate=(0, 0),
        *,
        axis_labels=None,
        rotate=None,
        shear=None,
        ndim=None,
        name=None,
        units=None,
    ) -> None:
        super().__init__(
            scale,
            translate,
            axis_labels=axis_labels,
            rotate=rotate,
            shear=shear,
            ndim=ndim,
            name=name,
            units=units,
        )
        if ndim is None:
            ndim = infer_ndim(
                scale=scale, translate=translate, rotate=rotate, shear=shear
            )
        self._translate = translate_to_vector(translate, ndim=ndim)
        self._scale = scale_to_vector(scale, ndim=ndim)
        self._rotate = rotate_to_matrix(rotate, ndim=ndim)
        self._shear = shear_to_matrix(shear, ndim=ndim)
        self._linear_matrix = self._make_linear_matrix()

    @property
    def scale(self) -> npt.NDArray:
        """Return the scale of the transform."""
        return self._scale

    @scale.setter
    def scale(self, scale):
        """Set the scale of the transform."""
        self._scale = scale_to_vector(scale, ndim=self.ndim)
        self._linear_matrix = self._make_linear_matrix()
        self._clean_cache()

    @property
    def rotate(self) -> npt.NDArray:
        """Return the rotation of the transform."""
        return self._rotate

    @rotate.setter
    def rotate(self, rotate):
        """Set the rotation of the transform."""
        self._rotate = rotate_to_matrix(rotate, ndim=self.ndim)
        self._linear_matrix = self._make_linear_matrix()
        self._clean_cache()

    @property
    def shear(self) -> npt.NDArray:
        """Return the shear of the transform."""
        return (
            self._shear[np.triu_indices(n=self.ndim, k=1)]
            if is_matrix_upper_triangular(self._shear)
            else self._shear
        )

    @shear.setter
    def shear(self, shear):
        """Set the shear of the transform."""
        self._shear = shear_to_matrix(shear, ndim=self.ndim)
        self._linear_matrix = self._make_linear_matrix()
        self._clean_cache()

    @property
    def linear_matrix(self):
        return super().linear_matrix

    @linear_matrix.setter
    def linear_matrix(self, linear_matrix):
        """Setting the linear matrix of a CompositeAffine transform is not supported."""
        raise NotImplementedError(
            trans._(
                'linear_matrix cannot be set directly for a CompositeAffine transform',
                deferred=True,
            )
        )

    @property
    def affine_matrix(self):
        return super().affine_matrix

    @affine_matrix.setter
    def affine_matrix(self, affine_matrix):
        """Setting the affine matrix of a CompositeAffine transform is not supported."""
        raise NotImplementedError(
            trans._(
                'affine_matrix cannot be set directly for a CompositeAffine transform',
                deferred=True,
            )
        )

    def set_slice(self, axes: Sequence[int]) -> 'CompositeAffine':
        return CompositeAffine(
            scale=self._scale[axes],
            translate=self._translate[axes],
            rotate=self._rotate[np.ix_(axes, axes)],
            shear=self._shear[np.ix_(axes, axes)],
            ndim=len(axes),
            name=self.name,
            units=[self.units[i] for i in axes],
            axis_labels=[self.axis_labels[i] for i in axes],
        )

    def expand_dims(self, axes: Sequence[int]) -> 'CompositeAffine':
        n = len(axes) + len(self.scale)
        not_axes = [i for i in range(n) if i not in axes]
        rotate = np.eye(n)
        rotate[np.ix_(not_axes, not_axes)] = self._rotate
        shear = np.eye(n)
        shear[np.ix_(not_axes, not_axes)] = self._shear
        translate = np.zeros(n)
        translate[not_axes] = self._translate
        scale = np.ones(n)
        scale[not_axes] = self._scale
        return CompositeAffine(
            translate=translate,
            scale=scale,
            rotate=rotate,
            shear=shear,
            ndim=n,
            name=self.name,
        )

    def _make_linear_matrix(self):
        return self._rotate @ self._shear @ np.diag(self._scale)
