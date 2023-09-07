import numpy as np
from vispy.scene.visuals import Arrow, Compound, Text


class Measure(Compound):
    def __init__(self) -> None:
        super().__init__(
            [
                Arrow(np.zeros((0, 3)), color='red', antialias=True),
                Text('', color='red', anchor_y='bottom'),
            ]
        )

    @property
    def arrow(self):
        return self._subvisuals[0]

    @property
    def text(self):
        return self._subvisuals[1]
