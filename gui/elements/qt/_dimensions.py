from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QWidget, QSlider, QGridLayout

class QtDimensions(QWidget):
    def __init__(self, dimensions):
        super().__init__()

        self.dimensions = dimensions
        self.sliders = []

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setFixedHeight(0)

        self.dimensions.events.update_slider.connect(self.update_slider)

    def _axis_to_row(self, axis, dims):
        message = f'axis {axis} out of bounds for {dims} dims'

        if axis < 0:
            axis = dims - axis
            if axis < 0:
                raise IndexError(message)
        elif axis >= dims:
            raise IndexError(message)

        if axis < 2:
            raise ValueError('cannot convert y/x-axes to rows')

        return axis - 2

    def update_slider(self, event):
        """Updates a slider for the given axis or creates
        it if it does not already exist.

        Parameters
        ----------
        axis : int
            Axis that this slider controls.
        max_axis_length : int
            Longest length for this axis. If 0, deletes the slider.

        Returns
        -------
        slider : PyQt5.QSlider or None
            Updated slider, if it exists.
        """
        axis = event.dim
        max_axis_length = event.dim_len
        dims = self.dimensions.max_dims

        grid = self.layout()
        row = self._axis_to_row(axis, dims)

        slider = grid.itemAt(row)
        if max_axis_length <= 0:
            # delete slider
            grid.takeAt(row)
            return

        if slider is None:  # has not been created yet
            # create slider
            if axis < 0:
                raise ValueError('cannot create a slider '
                                 f'at negative axis {axis}')

            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.StrongFocus)
            slider.setMinimum(0)
            slider.setFixedHeight(17)
            slider.setTickPosition(QSlider.NoTicks)
            # slider.setTickPosition(QSlider.TicksBothSides)
            # tick_interval = int(max(8,max_axis_length/8))
            # slider.setTickInterval(tick_interval)
            slider.setSingleStep(1)

            grid.addWidget(slider, row, 0)
            self.sliders.append(slider)
        else:
            slider = slider.widget()

        slider.valueChanged.connect(lambda value, axis=axis: self._slider_value_changed(value, axis))
        slider.setMaximum(max_axis_length - 1)
        self.setFixedHeight((dims-2)*19)
        return slider

    def _slider_value_changed(self, value, axis):
        self.dimensions.indices[axis] = value
        self.dimensions._need_redraw = True
        self.dimensions._update()
