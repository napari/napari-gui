from vispy.scene.node import Node

from napari._vispy.layers.base import VispyBaseLayer


class VispyLayerGroup(VispyBaseLayer):
    def __init__(self, layer):
        self.node = Node()
        super().__init__(layer, self.node)

        self.layer.events.inserted.connect(self._on_inserted)

    def _on_inserted(self):
        # TODO: self.child.node.parent = self.node
        #       need access to both self.node and self.child.node!
        pass

    def _on_data_change(self):
        pass
