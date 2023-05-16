
import napari
from napari._qt.containers import QtLayerTreeView
from napari.layers import Points
from napari.layers.layergroup import LayerGroup
from napari.qt import get_app, run

app = get_app()

tip = Points(name='tip')
p2 = Points(name='2')
lg2 = LayerGroup([p2], name="g2")
lg1 = LayerGroup([lg2, Points(name='3'), tip, Points(name='1')], name="g1")
root = LayerGroup(
    [
        lg1,
        Points(name='4'),
        Points(name='5'),
        # Points(name='6'),
        # Points(name='7'),
        # Points(name='8'),
        # Points(name='9'),
    ],
    name="root",
)
root.events.reordered.connect(lambda e: print(e.value))
root.selection.active = root[0]

tree = QtLayerTreeView(root)

# spy on events
root.events.reordered.connect(lambda e: print("reordered to: ", e.value))
root.selection.events.changed.connect(
    lambda e: print(
        f"selection changed.  added: {e.added}, removed: {e.removed}"
    )
)
root.selection.events._current.connect(
    lambda e: print(f"current item changed to: {e.value}")
)

# model = tree.model()
# tree.show()

# im, pt = Image(grass()), Points()
# root2 = LayerGroup([LayerGroup([im, pt])])
# tree2 = QtLayerTreeView(root2)
v = napari.Viewer()

# v.layers.extend([im, pt])
v.layers.append(tip)
v.layers.append(p2)
v.window.add_dock_widget(tree, area='right')

run()
