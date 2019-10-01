# napari

### multi-dimensional image viewer for python

[![License](https://img.shields.io/pypi/l/napari.svg)](https://github.com/napari/napari/raw/master/LICENSE)
[![Build Status](https://api.cirrus-ci.com/github/Napari/napari.svg)](https://cirrus-ci.com/napari/napari)
[![codecov](https://codecov.io/gh/napari/napari/branch/master/graph/badge.svg)](https://codecov.io/gh/napari/napari)
[![Python Version](https://img.shields.io/pypi/pyversions/napari.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/napari.svg)](https://pypi.org/project/napari)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/napari.svg)](https://pypistats.org/packages/napari)
[![Development Status](https://img.shields.io/pypi/status/napari.svg)](https://github.com/napari/napari)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

**napari** is a fast, interactive, multi-dimensional image viewer for Python. It's designed for browsing, annotating, and analyzing large multi-dimensional images. It's built on top of `Qt` (for the GUI), `vispy` (for performant GPU-based rendering), and the scientific Python stack (`numpy`, `scipy`).

We're developing **napari** in the open! But the project is in an **alpha** stage, and there will still likely be **breaking changes** with each release. You can follow progress on this repository, test out new versions as we release them, and contribute ideas and code.

We're working on [in-depth tutorials](https://napari.github.io/napari-tutorials/), but you can also quickly get started by looking below.

## installation

**napari** can be installed on most macOS and Linux systems with Python 3.6 or 3.7 by calling

```sh
$ pip install napari
```

We're working on improving our Windows support.

To clone the repository locally and install in editable mode use

```sh
$ git clone https://github.com/napari/napari.git
$ cd napari
$ pip install -e .
```

For more information see our [installation tutorial](https://napari.github.io/napari-tutorials/tutorials/installation)

## simple example

From inside an IPython shell (started with `ipython --gui=qt`) or jupyter notebook (after running a cell with magic command `%gui qt5`) you can open up an interactive viewer by calling

```python
from skimage import data
import napari
viewer = napari.view_image(data.astronaut(), rgb=True)
```

![image](resources/screenshot-add-image.png)

To do the same thing inside a script call

```python
from skimage import data
import napari

with napari.gui_qt():
    viewer = napari.view_image(data.astronaut(), rgb=True)
```

## features

Check out the scripts in our `examples` folder to see some of the functionality we're developing!

**napari** supports six main different layer types, `Image`, `Labels`, `Points`, `Vectors`, `Shapes`, and `Surface`, each corresponding to a different data type, visualization, and interactivity. You can add multiple layers of different types into the viewer and then start working with them, adjusting their properties.

All our layer types support n-dimensional data and the viewer provides the ability to quickly browse and visualize either 2D or 3D slices of the data.

**napari** also supports bidirectional communication between the viewer and the Python kernel, which is especially useful when launching from jupyter notebooks or when using our built-in console. Using the console allows you to interactively load and save data from the viewer and control all the features of the viewer programmatically.

You can extend **napari** using custom shortcuts, key bindings, and mouse functions.

## tutorials

For more details on how to use `napari` checkout our [in-depth tutorials](https://napari.github.io/napari-tutorials/). These are still a work in progress, but we'll be updating them regularly.

## plans

We're working on several features, including

- support for multiple linked canvases
- script & macro generation
- a plugin ecosystem for integrating image processing and machine learning tools

See [this issue](https://github.com/napari/napari/issues/420) for some of the features on the roadmap for our `0.3` release. Feel free to add comments or ideas!

## contributing

Contributions are encouraged! Please read our [contributing guide](CONTRIBUTING.md) to get started. Given that we're in an early stage, you may want to reach out on our [Github Issues](https://github.com/napari/napari/issues) before jumping in.

## help

We're a community partner on the [imagesc forum](https://forum.image.sc/tags/napari) and all help and support requests should be posted on the forum with the tag `napari`. We look forward to interacting with you there.
