import numpy as np
from vispy.scene.visuals import Compound, Line, Rectangle, Text


class ScaleBar(Compound):
    def __init__(self) -> None:
        self._data = np.array(
            [
                [0, 0],
                [1, 0],
            ]
        )

        # order matters (last is drawn on top)
        super().__init__(
            [
                Rectangle(center=[0.5, 0.5], width=1.1, height=36),
                Text(
                    text='1px',
                    pos=[0.5, 0.5],
                    anchor_x='center',
                    anchor_y='top',
                    font_size=10,
                ),
                Line(connect='strip', method='gl', width=3),
            ]
        )

    @property
    def line(self):
        return self._subvisuals[2]

    @property
    def text(self):
        return self._subvisuals[1]

    @property
    def box(self):
        return self._subvisuals[0]

    def set_data(self, color, ticks):
        data = self._data if ticks else self._data[:2]
        self.line.set_data(data, color)
        self.text.color = color
