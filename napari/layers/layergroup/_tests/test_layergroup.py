from textwrap import dedent

import numpy as np
import pytest

from napari.layers import Image
from napari.layers.layergroup import LayerGroup


@pytest.fixture
def layergroup_tree(monkeypatch):
    """Build a layergroup to use for testing.

    tree
      ├──branch1
      │  ├──image1_on_branch1
      │  └──image2_on_branch1
      ├──branch2
      │  └──image1_on_branch2
      ├──branch3
      │  ├──branch4
      │  │  ├──image1_on_branch4_on_branch3
      │  │  └──image2_on_branch4_on_branch3
      │  ├──branch5
      │  │  └──image1_on_branch5_on_branch3
      │  └──image1_on_branch3
      └──single_img
    """
    monkeypatch.setenv('ALLOW_LAYERGROUPS', "1")
    data = np.random.random((100, 100))
    branch1 = LayerGroup(name='branch1')
    branch1.append(Image(data, name='image1_on_branch1'))
    branch1.append(Image(data, name='image2_on_branch1'))
    branch2 = LayerGroup(name='branch2')
    branch2.append(Image(data, name='image1_on_branch2'))
    branch3 = LayerGroup(name='branch3')
    branch4 = LayerGroup(name='branch4')
    branch4.append(Image(data, name='image1_on_branch4_on_branch3'))
    branch4.append(Image(data, name='image2_on_branch4_on_branch3'))
    branch5 = LayerGroup(
        [Image(data, name='image1_on_branch5_on_branch3')], name='branch5'
    )
    branch3.extend([branch4, branch5, Image(data, name='image1_on_branch3')])

    tree = LayerGroup(name='tree')
    tree.extend([branch1, branch2, branch3, Image(data, name='single_img')])
    return tree


def test_construct_layergroup(layergroup_tree: LayerGroup):
    """Test that basic construction of a layergroup works."""
    assert len(layergroup_tree) == 4
    assert len(list(layergroup_tree.traverse())) == 14
    assert len(list(layergroup_tree.traverse(leaves_only=True))) == 8

    expected_layergroup_string = dedent(
        """
        tree
          ├──branch1
          │  ├──image1_on_branch1
          │  └──image2_on_branch1
          ├──branch2
          │  └──image1_on_branch2
          ├──branch3
          │  ├──branch4
          │  │  ├──image1_on_branch4_on_branch3
          │  │  └──image2_on_branch4_on_branch3
          │  ├──branch5
          │  │  └──image1_on_branch5_on_branch3
          │  └──image1_on_branch3
          └──single_img
    """
    ).strip()

    assert str(layergroup_tree) == expected_layergroup_string
