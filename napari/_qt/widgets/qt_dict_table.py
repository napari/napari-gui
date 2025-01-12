import re

from qtpy.QtCore import QSize, Slot
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem

from napari.utils.translations import trans

email_pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}'
    r'\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
)


class QtDictTable(QTableWidget):
    """A QTableWidget subclass that makes a table from a list of dicts.

    This will also make any cells that contain emails address or URLs
    clickable to open the link in a browser/email client.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget, by default None
    source : list of dict, optional
        A list of dicts where each dict in the list is a row, and each key in
        the dict is a header, by default None.  (call set_data later to add
        data)
    headers : list of str, optional
        If provided, will be used in order as the headers of the table.  All
        items in ``headers`` must be present in at least one of the dicts.
        by default headers will be the set of all keys in all dicts in
        ``source``
    min_section_width : int, optional
        If provided, sets a minimum width on the columns, by default None
    max_section_width : int, optional
        Sets a maximum width on the columns, by default 480

    Raises
    ------
    ValueError
        if ``source`` is not a list of dicts.
    """

    def __init__(
        self,
        parent=None,
        source: list[dict] | None = None,
        *,
        headers: list[str] | None = None,
        min_section_width: int | None = None,
        max_section_width: int = 480,
    ) -> None:
        super().__init__(parent=parent)
        if min_section_width:
            self.horizontalHeader().setMinimumSectionSize(min_section_width)
        self.horizontalHeader().setMaximumSectionSize(max_section_width)
        self.horizontalHeader().setStretchLastSection(True)
        if source:
            self.set_data(source, headers)
        self.cellClicked.connect(self._go_to_links)
        self.setMouseTracking(True)

    def set_data(self, data: list[dict], headers: list[str] | None = None):
        """Set the data in the table, given a list of dicts.

        Parameters
        ----------
        data : List[dict]
            A list of dicts where each dict in the list is a row, and each key
            in the dict is a header, by default None.  (call set_data later to
            add data)
        headers : list of str, optional
            If provided, will be used in order as the headers of the table. All
            items in ``headers`` must be present in at least one of the dicts.
            by default headers will be the set of all keys in all dicts in
            ``source``
        """
        if not isinstance(data, list) or any(
            not isinstance(i, dict) for i in data
        ):
            raise ValueError(
                trans._(
                    "'data' argument must be a list of dicts", deferred=True
                )
            )
        nrows = len(data)
        _headers = sorted(set().union(*data))
        if headers:
            for h in headers:
                if h not in _headers:
                    raise ValueError(
                        trans._(
                            "Argument 'headers' got item '{header}', which was not found in any of the items in 'data'",
                            deferred=True,
                            header=h,
                        )
                    )
            _headers = headers
        self.setRowCount(nrows)
        self.setColumnCount(len(_headers))
        for row, elem in enumerate(data):
            for key, value in elem.items():
                value = value or ''
                try:
                    col = _headers.index(key)
                except ValueError:
                    continue
                item = QTableWidgetItem(value)
                # underline links
                if email_pattern.match(value) or url_pattern.match(value):
                    font = QFont()
                    font.setUnderline(True)
                    item.setFont(font)
                self.setItem(row, col, item)

        self.setHorizontalHeaderLabels(_headers)
        self.resize_to_fit()

    @Slot(int, int)
    def _go_to_links(self, row, col):
        """if a cell is clicked and it contains an email or url, go to link."""
        import webbrowser

        item = self.item(row, col)
        text = item.text().strip()
        if email_pattern.match(text):
            webbrowser.open(f'mailto:{text}', new=1)
            return
        if url_pattern.match(text):
            webbrowser.open(text, new=1)

    def resize_to_fit(self):
        self.resizeColumnsToContents()
        self.resize(self.sizeHint())

    def sizeHint(self):
        """Return (width, height) of the table"""
        width = sum(map(self.columnWidth, range(self.columnCount()))) + 25
        height = self.rowHeight(0) * (self.rowCount() + 1)
        return QSize(width, height)
