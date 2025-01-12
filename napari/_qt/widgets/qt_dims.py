import warnings

import numpy as np
from qtpy.QtCore import Slot
from qtpy.QtGui import QFont, QFontMetrics
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from napari._qt.widgets.qt_dims_slider import (
    AnimationThread,
    QtDimSliderWidget,
)
from napari.components.dims import Dims
from napari.settings._constants import LoopMode
from napari.utils.translations import trans


class QtDims(QWidget):
    """Qt view for the napari Dims model.

    Parameters
    ----------
    dims : napari.components.dims.Dims
        Dims object to be passed to Qt object.
    parent : QWidget, optional
        QWidget that will be the parent of this widget.

    Attributes
    ----------
    dims : napari.components.dims.Dims
        Dimensions object modeling slicing and displaying.
    slider_widgets : list[QtDimSliderWidget]
        List of slider widgets.
    """

    def __init__(self, dims: Dims, parent=None) -> None:
        super().__init__(parent=parent)

        self.SLIDERHEIGHT = 22

        # We keep a reference to the view:
        self.dims = dims

        # list of sliders
        self.slider_widgets = []

        # True / False if slider is or is not displayed
        self._displayed_sliders = []

        self._animation_thread = AnimationThread(self)

        # Initialises the layout:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Update the number of sliders now that the dims have been added
        self._update_nsliders()
        self.dims.events.ndim.connect(self._update_nsliders)
        self.dims.events.current_step.connect(self._update_slider)
        self.dims.events.range.connect(self._update_range)
        self.dims.events.ndisplay.connect(self._update_display)
        self.dims.events.order.connect(self._update_display)
        self.dims.events.last_used.connect(self._on_last_used_changed)

    @property
    def nsliders(self):
        """Returns the number of sliders.

        Returns
        -------
        nsliders: int
            Number of sliders.
        """
        return len(self.slider_widgets)

    def _on_last_used_changed(self):
        """Sets the style of the last used slider."""
        for i, widget in enumerate(self.slider_widgets):
            sld = widget.slider
            sld.setProperty('last_used', i == self.dims.last_used)
            sld.style().unpolish(sld)
            sld.style().polish(sld)

    def _update_slider(self):
        """Updates position for a given slider."""
        for widget in self.slider_widgets:
            widget._update_slider()

    def _update_range(self):
        """Updates range for a given slider."""
        for widget in self.slider_widgets:
            widget._update_range()

        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_slice_labels()

    def _update_display(self):
        """Updates display for all sliders."""
        widgets = reversed(list(enumerate(self.slider_widgets)))
        nsteps = self.dims.nsteps
        for axis, widget in widgets:
            if axis in self.dims.displayed or nsteps[axis] <= 1:
                # Displayed dimensions correspond to non displayed sliders
                self._displayed_sliders[axis] = False
                self.dims.last_used = 0
                widget.hide()
            else:
                # Non displayed dimensions correspond to displayed sliders
                self._displayed_sliders[axis] = True
                self.dims.last_used = axis
                widget.show()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_slice_labels()
        self._resize_axis_labels()
        self.stop()

    def _update_nsliders(self):
        """Updates the number of sliders based on the number of dimensions."""
        self._trim_sliders(0)
        self._create_sliders(self.dims.ndim)
        self._update_display()
        for i in range(self.dims.ndim):
            self._update_range()
            if self._displayed_sliders[i]:
                self._update_slider()
        self.stop()

    def _resize_axis_labels(self):
        """When any of the labels get updated, this method updates all label
        widths to a minimum size. This allows the full label to be
        visible at all times, with minimal space, without setting stretch on
        the layout.
        """
        displayed_labels = [
            self.slider_widgets[idx].axis_label
            for idx, displayed in enumerate(self._displayed_sliders)
            if displayed
        ]
        if displayed_labels:
            fm = self.fontMetrics()
            # set maximum width to no more than 20% of slider width
            maxwidth = int(self.slider_widgets[0].width() * 0.2)
            # set new width to the width of the longest label being displayed
            newwidth = max(
                [
                    int(fm.boundingRect(dlab.text()).width())
                    for dlab in displayed_labels
                ]
            )

            for slider in self.slider_widgets:
                labl = slider.axis_label
                # here the average width of a character is used as base measure
                # to add some extra width. We use 4 to take into account a
                # space and the possible 3 dots (`...`) for elided text
                margin_width = int(fm.averageCharWidth() * 4)
                base_labl_width = min([newwidth, maxwidth])
                labl_width = base_labl_width + margin_width
                labl.setFixedWidth(labl_width)

    def _resize_slice_labels(self):
        """When the size of any dimension changes, we want to resize all of the
        slice labels to width of the longest label, to keep all the sliders
        right aligned.  The width is determined by the number of digits in the
        largest dimensions, plus a little padding.
        """
        width = 0
        for ax, maxi in enumerate(self.dims.nsteps):
            if self._displayed_sliders[ax]:
                length = len(str(maxi - 1))
                if length > width:
                    width = length
        # gui width of a string of length `width`
        fm = QFontMetrics(QFont('', 0))
        width = fm.boundingRect('8' * width).width()
        for labl in self.findChildren(QWidget, 'slice_label'):
            labl.setFixedWidth(width + 6)

    def _create_sliders(self, number_of_sliders: int):
        """Creates sliders to match new number of dimensions.

        Parameters
        ----------
        number_of_sliders : int
            New number of sliders.
        """
        # add extra sliders so that number_of_sliders are present
        # add to the beginning of the list
        for slider_num in range(self.nsliders, number_of_sliders):
            dim_axis = number_of_sliders - slider_num - 1
            slider_widget = QtDimSliderWidget(self, dim_axis)
            slider_widget.axis_label.textChanged.connect(
                self._resize_axis_labels
            )
            slider_widget.size_changed.connect(self._resize_axis_labels)
            slider_widget.play_button.play_requested.connect(self.play)
            self.layout().addWidget(slider_widget)
            self.slider_widgets.insert(0, slider_widget)
            self._displayed_sliders.insert(0, True)
            nsliders = np.sum(self._displayed_sliders)
            self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_axis_labels()

    def _trim_sliders(self, number_of_sliders):
        """Trims number of dimensions to a lower number.

        Parameters
        ----------
        number_of_sliders : int
            New number of sliders.
        """
        # remove extra sliders so that only number_of_sliders are left
        # remove from the beginning of the list
        for _slider_num in range(number_of_sliders, self.nsliders):
            self._remove_slider_widget(0)

    def _remove_slider_widget(self, index):
        """Remove slider_widget at index, including all sub-widgets.

        Parameters
        ----------
        index : int
            Index of slider to remove
        """
        # remove particular slider
        slider_widget = self.slider_widgets.pop(index)
        self._displayed_sliders.pop(index)
        self.layout().removeWidget(slider_widget)
        # As we delete this widget later, callbacks with a weak reference
        # to it may successfully grab the instance, but may be incompatible
        # with other update state like dims.
        self.dims.events.axis_labels.disconnect(slider_widget._pull_label)
        slider_widget.deleteLater()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(int(nsliders * self.SLIDERHEIGHT))
        self.dims.last_used = 0

    def play(
        self,
        axis: int = 0,
        fps: float | None = None,
        loop_mode: str | None = None,
        frame_range: tuple[int, int] | None = None,
    ):
        """Animate (play) axis.

        Parameters
        ----------
        axis : int
            Index of axis to play
        fps : float
            Frames per second for playback.  Negative values will play in
            reverse.  fps == 0 will stop the animation. The view is not
            guaranteed to keep up with the requested fps, and may drop frames
            at higher fps.
        loop_mode : str
            Mode for animation playback.  Must be one of the following options:
                "once": Animation will stop once movie reaches the
                    max frame (if fps > 0) or the first frame (if fps < 0).
                "loop":  Movie will return to the first frame
                    after reaching the last frame, looping until stopped.
                "back_and_forth":  Movie will loop back and forth until
                    stopped
        frame_range : tuple | list
            If specified, will constrain animation to loop [first, last] frames

        Raises
        ------
        IndexError
            If ``axis`` requested is out of the range of the dims
        IndexError
            If ``frame_range`` is provided and out of the range of the dims
        ValueError
            If ``frame_range`` is provided and range[0] >= range[1]
        """
        # doing manual check here to avoid issue in StringEnum
        # see https://github.com/napari/napari/issues/754
        if loop_mode is not None:
            _modes = LoopMode.keys()
            if loop_mode not in _modes:
                raise ValueError(
                    trans._(
                        'loop_mode must be one of {_modes}. Got: {loop_mode}',
                        _modes=_modes,
                        loop_mode=loop_mode,
                    )
                )
            loop_mode = LoopMode(loop_mode)

        if axis >= self.dims.ndim:
            raise IndexError(trans._('axis argument out of range'))

        if self.is_playing and self._animation_thread.axis == axis:
            self.slider_widgets[axis]._update_play_settings(
                fps, loop_mode, frame_range
            )
            return

        # we want to avoid playing a dimension that does not have a slider
        # (like X or Y, or a third dimension in volume view.)
        if self._displayed_sliders[axis]:
            if self._animation_thread.isRunning():
                self._animation_thread.slider.play_button._handle_stop()
            self.slider_widgets[axis]._update_play_settings(
                fps, loop_mode, frame_range
            )
            self._animation_thread.set_slider(self.slider_widgets[axis])
            self._animation_thread.frame_requested.connect(self._set_frame)
            if not self._animation_thread.isRunning():
                self._animation_thread.start()
            else:
                self._animation_thread.slider.play_button._handle_start()

        else:
            warnings.warn(
                trans._(
                    'Refusing to play a hidden axis',
                    deferred=True,
                )
            )

    @Slot()
    def stop(self):
        """Stop axis animation"""
        self._animation_thread._stop()

    @property
    def is_playing(self):
        """Return True if any axis is currently animated."""
        try:
            return not self._animation_thread._waiter.is_set()
        except RuntimeError as e:  # pragma: no cover
            if (
                'wrapped C/C++ object of type' not in e.args[0]
                and 'Internal C++ object' not in e.args[0]
            ):
                # checking if threat is partially deleted. Otherwise
                # reraise exception. For more details see:
                # https://github.com/napari/napari/pull/5499
                raise
            return False

    def _set_frame(self, axis, frame):
        """Safely tries to set `axis` to the requested `point`.

        This function is debounced: if the previous frame has not yet drawn to
        the canvas, it will simply do nothing.  If the timer plays faster than
        the canvas can draw, this will drop the intermediate frames, keeping
        the effective frame rate constant even if the canvas cannot keep up.
        """
        if self.dims._play_ready:
            # disable additional point advance requests until this one draws
            self.dims._play_ready = False
            self.dims.set_current_step(axis, frame)

    def closeEvent(self, event):
        [w.deleteLater() for w in self.slider_widgets]
        self.deleteLater()
        event.accept()
