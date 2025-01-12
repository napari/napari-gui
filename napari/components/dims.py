from collections.abc import Sequence
from numbers import Integral
from typing import (
    Any,
    Literal,
    NamedTuple,
)

import numpy as np

from napari._pydantic_compat import root_validator, validator
from napari.utils.events import EventedModel
from napari.utils.misc import argsort, reorder_after_dim_reduction
from napari.utils.translations import trans


class RangeTuple(NamedTuple):
    start: float
    stop: float
    step: float


class Dims(EventedModel):
    """Dimensions object modeling slicing and displaying.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    ndisplay : int
        Number of displayed dimensions.
    range : tuple of 3-tuple of float
        List of tuples (min, max, step), one for each dimension in world
        coordinates space. Lower and upper bounds are inclusive.
    point : tuple of floats
        Dims position in world coordinates for each dimension.
    margin_left : tuple of floats
        Left margin in world pixels of the slice for each dimension.
    margin_right : tuple of floats
        Right margin in world pixels of the slice for each dimension.
    order : tuple of int
        Tuple of ordering the dimensions, where the last dimensions are rendered.
    axis_labels : tuple of str
        Tuple of labels for each dimension.
    last_used : int
        Dimension which was last interacted with.

    Attributes
    ----------
    ndim : int
        Number of dimensions.
    ndisplay : int
        Number of displayed dimensions.
    range : tuple of 3-tuple of float
        List of tuples (min, max, step), one for each dimension in world
        coordinates space. Lower and upper bounds are inclusive.
    point : tuple of floats
        Dims position in world coordinates for each dimension.
    margin_left : tuple of floats
        Left margin (=thickness) in world pixels of the slice for each dimension.
    margin_right : tuple of floats
        Right margin (=thickness) in world pixels of the slice for each dimension.
    order : tuple of int
        Tuple of ordering the dimensions, where the last dimensions are rendered.
    axis_labels : tuple of str
        Tuple of labels for each dimension.
    last_used : int
        Dimension which was last used.
        Tuple the slider position for each dims slider, in world coordinates.
    current_step : tuple of int
        Current step for each dimension (same as point, but in slider coordinates).
    nsteps : tuple of int
        Number of steps available to each slider. These are calculated from
        the ``range``.
    thickness : tuple of floats
        Thickness of the slice (sum of both margins) for each dimension in world coordinates.
    displayed : tuple of int
        List of dimensions that are displayed. These are calculated from the
        ``order`` and ``ndisplay``.
    not_displayed : tuple of int
        List of dimensions that are not displayed. These are calculated from the
        ``order`` and ``ndisplay``.
    displayed_order : tuple of int
        Order of only displayed dimensions. These are calculated from the
        ``displayed`` dimensions.
    rollable :  tuple of bool
        Tuple of axis roll state. If True the axis is rollable.
    """

    # fields
    ndim: int = 2
    ndisplay: Literal[2, 3] = 2

    order: tuple[int, ...] = ()
    axis_labels: tuple[str, ...] = ()
    rollable: tuple[bool, ...] = ()

    range: tuple[RangeTuple, ...] = ()
    margin_left: tuple[float, ...] = ()
    margin_right: tuple[float, ...] = ()
    point: tuple[float, ...] = ()

    last_used: int = 0

    # private vars
    _play_ready: bool = True  # False if currently awaiting a draw event
    _scroll_progress: int = 0

    # validators
    # check fields is false to allow private fields to work
    @validator(
        'order',
        'axis_labels',
        'rollable',
        'point',
        'margin_left',
        'margin_right',
        pre=True,
        allow_reuse=True,
    )
    def _as_tuple(v):
        return tuple(v)

    @validator('range', pre=True)
    def _check_ranges(ranges):
        """
        Ensure the range values are sane.

        - start < stop
        - step > 0
        """
        for axis, (start, stop, step) in enumerate(ranges):
            if start > stop:
                raise ValueError(
                    trans._(
                        'start and stop must be strictly increasing, but got ({start}, {stop}) for axis {axis}',
                        deferred=True,
                        start=start,
                        stop=stop,
                        axis=axis,
                    )
                )
            if step <= 0:
                raise ValueError(
                    trans._(
                        'step must be strictly positive, but got {step} for axis {axis}.',
                        deferred=True,
                        step=step,
                        axis=axis,
                    )
                )
        return ranges

    @root_validator(skip_on_failure=True, allow_reuse=True)
    def _check_dims(cls, values):
        """Check the consistency of dimensionality for all attributes.

        Parameters
        ----------
        values : dict
            Values dictionary to update dims model with.
        """
        updated = {}

        ndim = values['ndim']

        range_ = ensure_len(values['range'], ndim, pad_width=(0.0, 2.0, 1.0))
        updated['range'] = tuple(RangeTuple(*rng) for rng in range_)

        point = ensure_len(values['point'], ndim, pad_width=0.0)
        # ensure point is limited to range
        updated['point'] = tuple(
            np.clip(pt, rng.start, rng.stop)
            for pt, rng in zip(point, updated['range'], strict=False)
        )

        updated['margin_left'] = ensure_len(
            values['margin_left'], ndim, pad_width=0.0
        )
        updated['margin_right'] = ensure_len(
            values['margin_right'], ndim, pad_width=0.0
        )

        # order and label default computation is too different to include in ensure_len()
        # Check the order tuple has same number of elements as ndim
        order = values['order']
        if len(order) < ndim:
            order_ndim = len(order)
            # new dims are always prepended
            prepended_dims = tuple(range(ndim - order_ndim))
            # maintain existing order, but shift accordingly
            existing_order = tuple(o + ndim - order_ndim for o in order)
            order = prepended_dims + existing_order
        elif len(order) > ndim:
            order = reorder_after_dim_reduction(order[-ndim:])
        updated['order'] = order

        # Check the order is a permutation of 0, ..., ndim - 1
        if set(updated['order']) != set(range(ndim)):
            raise ValueError(
                trans._(
                    'Invalid ordering {order} for {ndim} dimensions',
                    deferred=True,
                    order=updated['order'],
                    ndim=ndim,
                )
            )

        # Check the axis labels tuple has same number of elements as ndim
        axis_labels = values['axis_labels']
        labels_ndim = len(axis_labels)
        if labels_ndim < ndim:
            # Append new "default" labels to existing ones
            if axis_labels == tuple(map(str, range(labels_ndim))):
                updated['axis_labels'] = tuple(map(str, range(ndim)))
            else:
                updated['axis_labels'] = (
                    tuple(map(str, range(ndim - labels_ndim))) + axis_labels
                )
        elif labels_ndim > ndim:
            updated['axis_labels'] = axis_labels[-ndim:]

        # Check the rollable axes tuple has same number of elements as ndim
        updated['rollable'] = ensure_len(values['rollable'], ndim, True)

        # If the last used slider is no longer visible, use the first.
        last_used = values['last_used']
        ndisplay = values['ndisplay']
        dims_range = updated['range']
        nsteps = cls._nsteps_from_range(dims_range)
        not_displayed = [
            d for d in order[:-ndisplay] if len(nsteps) > d and nsteps[d] > 1
        ]
        if len(not_displayed) > 0 and last_used not in not_displayed:
            updated['last_used'] = not_displayed[0]

        return {**values, **updated}

    @staticmethod
    def _nsteps_from_range(dims_range) -> tuple[float, ...]:
        return tuple(
            # "or 1" ensures degenerate dimension works
            int((rng.stop - rng.start) / (rng.step or 1)) + 1
            for rng in dims_range
        )

    @property
    def nsteps(self) -> tuple[float, ...]:
        return self._nsteps_from_range(self.range)

    @nsteps.setter
    def nsteps(self, value):
        self.range = tuple(
            RangeTuple(
                rng.start, rng.stop, (rng.stop - rng.start) / (nsteps - 1)
            )
            for rng, nsteps in zip(self.range, value, strict=False)
        )

    @property
    def current_step(self):
        return tuple(
            int(round((point - rng.start) / (rng.step or 1)))
            for point, rng in zip(self.point, self.range, strict=False)
        )

    @current_step.setter
    def current_step(self, value):
        self.point = tuple(
            rng.start + point * rng.step
            for point, rng in zip(value, self.range, strict=False)
        )

    @property
    def thickness(self) -> tuple[float, ...]:
        return tuple(
            left + right
            for left, right in zip(
                self.margin_left, self.margin_right, strict=False
            )
        )

    @thickness.setter
    def thickness(self, value):
        self.margin_left = self.margin_right = tuple(val / 2 for val in value)

    @property
    def displayed(self) -> tuple[int, ...]:
        """Tuple: Dimensions that are displayed."""
        return self.order[-self.ndisplay :]

    @property
    def not_displayed(self) -> tuple[int, ...]:
        """Tuple: Dimensions that are not displayed."""
        return self.order[: -self.ndisplay]

    @property
    def displayed_order(self) -> tuple[int, ...]:
        return tuple(argsort(self.displayed))

    def set_range(
        self,
        axis: int | Sequence[int],
        _range: Sequence[int | float] | Sequence[Sequence[int | float]],
    ):
        """Sets ranges (min, max, step) for the given dimensions.

        Parameters
        ----------
        axis : int or sequence of int
            Dimension index or a sequence of axes whos range will be set.
        _range : tuple or sequence of tuple
            Range specified as (min, max, step) or a sequence of these range
            tuples.
        """
        axis, value = self._sanitize_input(
            axis, _range, value_is_sequence=True
        )
        full_range = list(self.range)
        for ax, val in zip(axis, value, strict=False):
            full_range[ax] = val
        self.range = tuple(full_range)

    def set_point(
        self,
        axis: int | Sequence[int],
        value: float | Sequence[float],
    ):
        """Sets point to slice dimension in world coordinates.

        Parameters
        ----------
        axis : int or sequence of int
            Dimension index or a sequence of axes whos point will be set.
        value : scalar or sequence of scalars
            Value of the point for each axis.
        """
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        full_point = list(self.point)
        for ax, val in zip(axis, value, strict=False):
            full_point[ax] = val
        self.point = tuple(full_point)

    def set_current_step(
        self,
        axis: int | Sequence[int],
        value: int | Sequence[int],
    ):
        axis, value = self._sanitize_input(
            axis, value, value_is_sequence=False
        )
        range_ = list(self.range)
        value_world = []
        for ax, val in zip(axis, value, strict=False):
            rng = range_[ax]
            value_world.append(rng.start + val * rng.step)
        self.set_point(axis, value_world)

    def set_axis_label(
        self,
        axis: int | Sequence[int],
        label: str | Sequence[str],
    ):
        """Sets new axis labels for the given axes.

        Parameters
        ----------
        axis : int or sequence of int
            Dimension index or a sequence of axes whos labels will be set.
        label : str or sequence of str
            Given labels for the specified axes.
        """
        axis, label = self._sanitize_input(
            axis, label, value_is_sequence=False
        )
        full_axis_labels = list(self.axis_labels)
        for ax, val in zip(axis, label, strict=False):
            full_axis_labels[ax] = val
        self.axis_labels = tuple(full_axis_labels)

    def reset(self):
        """Reset dims values to initial states."""
        # Don't reset axis labels
        # TODO: could be optimized with self.update, but need to fix
        #       event firing in EventedModel first
        self.range = ((0, 2, 1),) * self.ndim
        self.point = (0,) * self.ndim
        self.order = tuple(range(self.ndim))
        self.margin_left = (0,) * self.ndim
        self.margin_right = (0,) * self.ndim
        self.rollable = (True,) * self.ndim

    def transpose(self):
        """Transpose displayed dimensions.

        This swaps the order of the last two displayed dimensions.
        The order of the displayed is taken from Dims.order.
        """
        order = list(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order

    def _increment_dims_right(self, axis: int | None = None):
        """Increment dimensions to the right along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        self.set_current_step(axis, self.current_step[axis] + 1)

    def _increment_dims_left(self, axis: int | None = None):
        """Increment dimensions to the left along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        self.set_current_step(axis, self.current_step[axis] - 1)

    def _focus_up(self):
        """Shift focused dimension slider to be the next slider above."""
        sliders = [d for d in self.not_displayed if self.nsteps[d] > 1]
        if len(sliders) == 0:
            return

        index = (sliders.index(self.last_used) + 1) % len(sliders)
        self.last_used = sliders[index]

    def _focus_down(self):
        """Shift focused dimension slider to be the next slider bellow."""
        sliders = [d for d in self.not_displayed if self.nsteps[d] > 1]
        if len(sliders) == 0:
            return

        index = (sliders.index(self.last_used) - 1) % len(sliders)
        self.last_used = sliders[index]

    def roll(self):
        """Roll order of dimensions for display."""
        order = np.array(self.order)
        # we combine "rollable" and "nsteps" into a mask for rolling
        # this mask has to be aligned to "order" as "rollable" and
        # "nsteps" are static but order is dynamic, meaning "rollable"
        # and "nsteps" encode the axes by position, whereas "order"
        # encodes axis by number
        valid = np.logical_and(self.rollable, np.array(self.nsteps) > 1)[order]
        order[valid] = np.roll(order[valid], shift=1)
        self.order = order

    def _go_to_center_step(self):
        self.current_step = [int((ns - 1) / 2) for ns in self.nsteps]

    def _sanitize_input(
        self, axis, value, value_is_sequence=False
    ) -> tuple[list[int], list]:
        """
        Ensure that axis and value are the same length, that axes are not
        out of bounds, and coerces to lists for easier processing.
        """
        if isinstance(axis, Integral):
            if (
                isinstance(value, Sequence)
                and not isinstance(value, str)
                and not value_is_sequence
            ):
                raise ValueError(
                    trans._('cannot set multiple values to a single axis')
                )
            axis = [axis]
            value = [value]
        else:
            axis = list(axis)
            value = list(value)

        if len(axis) != len(value):
            raise ValueError(
                trans._('axis and value sequences must have equal length')
            )

        for ax in axis:
            ensure_axis_in_bounds(ax, self.ndim)
        return axis, value


def ensure_len(value: tuple, length: int, pad_width: Any):
    """
    Ensure that the value has the required number of elements.

    Right-crop if value is too long; left-pad with default if too short.

    Parameters
    ----------
    value : Tuple
        A tuple of values to be resized.
    ndim : int
        Number of desired values.
    default : Tuple
        Default element for left-padding.
    """
    if len(value) < length:
        # left pad
        value = (pad_width,) * (length - len(value)) + value
    elif len(value) > length:
        # right-crop
        value = value[-length:]
    return value


def ensure_axis_in_bounds(axis: int, ndim: int) -> int:
    """Ensure a given value is inside the existing axes of the image.

    Returns
    -------
    axis : int
        The axis which was checked for validity.
    ndim : int
        The dimensionality of the layer.

    Raises
    ------
    ValueError
        The given axis index is out of bounds.
    """
    if axis not in range(-ndim, ndim):
        msg = trans._(
            'Axis {axis} not defined for dimensionality {ndim}. Must be in [{ndim_lower}, {ndim}).',
            deferred=True,
            axis=axis,
            ndim=ndim,
            ndim_lower=-ndim,
        )
        raise ValueError(msg)

    return axis % ndim
