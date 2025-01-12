"""
A Class for generating QIcons from SVGs with arbitrary colors at runtime.
"""

from qtpy.QtCore import QByteArray, QPoint, QRect, QRectF, Qt
from qtpy.QtGui import QIcon, QIconEngine, QImage, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer


class QColoredSVGIcon(QIcon):
    """A QIcon class that specializes in colorizing SVG files.

    Parameters
    ----------
    path_or_xml : str
        Raw SVG XML or a path to an existing svg file.  (Will raise error on
        ``__init__`` if a non-existent file is provided.)
    color : str, optional
        A valid CSS color string, used to colorize the SVG. by default None.
    opacity : float, optional
        Fill opacity for the icon (0-1).  By default 1 (opaque).

    Examples
    --------
    >>> from napari._qt.qt_resources import QColoredSVGIcon
    >>> from qtpy.QtWidgets import QLabel

    # Create icon with specific color
    >>> label = QLabel()
    >>> icon = QColoredSVGIcon.from_resources('new_points')
    >>> label.setPixmap(icon.colored('#0934e2', opacity=0.7).pixmap(300, 300))
    >>> label.show()

    # Create colored icon using theme
    >>> label = QLabel()
    >>> icon = QColoredSVGIcon.from_resources('new_points')
    >>> label.setPixmap(icon.colored(theme='light').pixmap(300, 300))
    >>> label.show()
    """

    def __init__(
        self,
        path_or_xml: str,
        color: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        from napari.resources import get_colorized_svg

        self._svg = path_or_xml
        colorized = get_colorized_svg(path_or_xml, color, opacity)
        super().__init__(SVGBufferIconEngine(colorized))

    def colored(
        self,
        color: str | None = None,
        opacity: float = 1.0,
        theme: str | None = None,
        theme_key: str = 'icon',
    ) -> 'QColoredSVGIcon':
        """Return a new colorized QIcon instance.

        Parameters
        ----------
        color : str, optional
            A valid CSS color string, used to colorize the SVG.  If provided,
            will take precedence over ``theme``, by default None.
        opacity : float, optional
            Fill opacity for the icon (0-1).  By default 1 (opaque).
        theme : str, optional
            Name of the theme to from which to get `theme_key` color.
            ``color`` argument takes precedence.
        theme_key : str, optional
            If using a theme, key in the theme dict to use, by default 'icon'

        Returns
        -------
        QColoredSVGIcon
            A pre-colored QColoredSVGIcon (which may still be recolored)
        """
        if not color and theme:
            from napari.utils.theme import get_theme

            color = getattr(get_theme(theme), theme_key).as_hex()

        return QColoredSVGIcon(self._svg, color, opacity)

    @staticmethod
    def from_resources(
        icon_name: str,
    ) -> 'QColoredSVGIcon':
        """Get an icon from napari SVG resources.

        Parameters
        ----------
        icon_name : str
            The name of the icon svg to load (just the stem).  Must be in the
            napari icons folder.

        Returns
        -------
        QColoredSVGIcon
            A colorizeable QIcon
        """
        from napari.resources import get_icon_path

        path = get_icon_path(icon_name)
        return QColoredSVGIcon(path)


class SVGBufferIconEngine(QIconEngine):
    """A custom QIconEngine that can render an SVG buffer.

    An icon engine provides the rendering functions for a ``QIcon``.
    Each icon has a corresponding icon engine that is responsible for drawing
    the icon with a requested size, mode and state.  While the built-in
    QIconEngine is capable of rendering SVG files, it's not able to receive the
    raw XML string from memory.

    This ``QIconEngine`` takes in SVG data as a raw xml string or bytes.

    see: https://doc.qt.io/qt-5/qiconengine.html
    """

    def __init__(self, xml: str | bytes) -> None:
        if isinstance(xml, str):
            xml = xml.encode('utf-8')
        self.data = QByteArray(xml)
        super().__init__()

    def paint(self, painter: QPainter, rect, mode, state):
        """Paint the icon int ``rect`` using ``painter``."""
        renderer = QSvgRenderer(self.data)
        renderer.render(painter, QRectF(rect))

    def clone(self):
        """Required to subclass abstract QIconEngine."""
        return SVGBufferIconEngine(self.data)

    def pixmap(self, size, mode, state):
        """Return the icon as a pixmap with requested size, mode, and state."""
        img = QImage(size, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        pixmap = QPixmap.fromImage(img, Qt.NoFormatConversion)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(QPoint(0, 0), size), mode, state)
        return pixmap
