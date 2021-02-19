from pathlib import Path

from qtpy.QtWidgets import QLabel

import napari
from napari.qt import compile_qt_resources


icon_path = Path(__file__).parent / 'grin.svg'  # icon svg in this folder
color = '#3956C9'

# colorize and compile our icons for use in the Gui.
# will generate all color-icon combinations provided
# using register=True will immediately register the resources for
# use with stylesheets and QIcons
compile_qt_resources(svg_paths=[icon_path], colors=[color], register=True)
# your resource(s) will be available at this url template:
url = f':/{color}/{icon_path.name}'

# use it in a stylesheet
lbl = QLabel()
lbl.setStyleSheet(f"image: url({url}); min-width: 200px; padding: 20px;")

v = napari.Viewer()
v.window.add_dock_widget(lbl, area='right')

napari.run()
