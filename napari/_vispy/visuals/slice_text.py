from vispy.scene.visuals import Compound, Rectangle, Text


class SliceText(Compound):
    def __init__(self) -> None:
        # order matters (last is drawn on top)
        super().__init__(
            [
                Rectangle(center=[0.5, 0.5], width=1.1, height=36),
                Text(
                    text='1px',
                    pos=[0.5, 0.5],
                    anchor_x='right',
                    anchor_y='top',
                    font_size=10,
                ),
            ]
        )

    @property
    def text(self) -> Text:
        return self._subvisuals[1]

    @property
    def box(self) -> Rectangle:
        return self._subvisuals[0]
