[![Unit tests](https://github.com/shubhamwagh/nerf-toy/actions/workflows/ci.yml/badge.svg)](https://github.com/shubhamwagh/nerf-toy/actions/workflows/ci.yml)
[![Build](https://github.com/shubhamwagh/nerf-toy/actions/workflows/python-publish.yml/badge.svg)](https://github.com/shubhamwagh/nerf-toy/actions/workflows/python-publish.yml)
[![Python Versions](https://img.shields.io/pypi/pyversions/nerf-toy.svg)](https://pypi.org/project/nerf-toy)
[![PyPI Version](https://img.shields.io/pypi/v/nerf-toy.svg)](https://pypi.org/project/nerf-toy)
[![PyPI status](https://img.shields.io/pypi/status/nerf-toy.svg)](https://pypi.python.org/project/nerf-toy)

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubhamwagh/nerf-toy/blob/main/notebook/demo.ipynb)<br>

<h1 align="center">
  <br>
  nerf-toy
  <br>
</h1>

<h4 align="center">A 2D toy illustration of Neural Radiance Fields</h4>


<p align="center">
  <a href="#description">Description</a> •
  <a href="#features">Features</a> •
  <a href="#examples">Examples</a> •
  <a href="#references">References</a> •
  <a href="#license">License</a>
</p>


<div align="center">

|                                                Original                                                |                                              No Mapping                                              | Basic Mapping                                                                                                |                                          Fourier Feature Mapping                                           |
|:------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|--------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------:|
| <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/data/lion_face.jpg" width="175"> | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/lion_no_mapping.gif" width="175"> | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/basic_lion_face.gif" width="175"> | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/rff_lion_face.gif" width="175"> |

</div>

## Description

**nerf-toy** is a 2D toy illustration of the [Neural Radiance Fields](http://www.matthewtancik.com/nerf). Memorizing a
2D image using an MLP is a nice problem to understand before going into solving the problem
NeRF tackles i.e. memorising a 3D scene given multiple view points.

The task is to produce the _(r, g, b)_ value of an image as a
function of _(x,y)_ image coordinates. Our model is a coordinate-based multilayer perceptron.

<p align="center">
    <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/network_diagram.png" height="200">
</p>

This is implemented using 2D convolutions with a kernel size of (1, 1), which act as "**pointwise**" convolutions. This
is equivalent to a densely connected multilayer perceptron for each coordinate.
Also, we use batchnorm to speed up convergence.

The model is trained with the following input mappings $\gamma (\mathbf{v})$ for comparison :

- No mapping: $\gamma(\mathbf{v})= \mathbf{v}$.

- Basic mapping: $\gamma(\mathbf{v})=\left[ \cos(2 \pi \mathbf{v}),\sin(2 \pi \mathbf{v}) \right]^\mathrm{T}$.

<!-- - Positional encoding: $\gamma(\mathbf{v})=\left[ \ldots, \cos(2 \pi \sigma^{j/m} \mathbf{v}),\sin(2 \pi \sigma^{j/m} \mathbf{v}), \ldots \right]^\mathrm{T}$ for $j = 0, \ldots, m-1$.  -->

- Gaussian Fourier feature mapping: $\gamma(\mathbf{v})=\left[ \cos(2 \pi \mathbf B \mathbf{v}), \sin(2 \pi \mathbf B \mathbf{v}) \right]^\mathrm{T}$,
  where each entry in $\mathbf B \in \mathbb R^{m \times d}$ is sampled from $\mathcal N(0,\sigma^2)$

## Features
- Transforms: **Basic**, **Positional Encoding** and **Gaussian Fourier Feature**
- Data loader for any input image, where filepath, image url or bytes can be passed as input
- [Keras](https://keras.io) based NeRF toy model, which can be customised based on number of layer and output channels
- Metrics: [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
  , [SSIM](https://en.wikipedia.org/wiki/Structural_similarity)
- Custom training callback: **PredictionVideoSaverCallback**, **PlotLossesAndMetricsCallback**
- Utility functions to read and manipulate image

## Examples
The [demo notebook](https://colab.research.google.com/github/shubhamwagh/nerf-toy/blob/main/notebook/demo.ipynb) demonstrates the core idea  with full model training from scratch.

<div align="center">

|                                               Original                                               |                                                No Mapping                                                 | Basic Mapping                                                                                              |                                         Fourier Feature Mapping                                          |
|:----------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------:|
| <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/data/nature.jpg" width="175">  | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/nature_none.gif" width="175">  | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/nature_basic.gif" width="175">  | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/nature_rff.gif" width="175">  |
| <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/data/kitchen.jpg" width="175"> | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/kitchen_none.gif" width="175"> | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/kitchen_basic.gif" width="175"> | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/kitchen_rff.gif" width="175"> |

</div>

It can be observed that directly regressing the _(x, y)_ image coordinates results in blurry reconstructions.

Instead, lifting the input pixel coordinates _(x, y)_ to higher dimensions via transformations (e.g. gaussian fourier feature) 
makes it easier for network to learn high frequency functions in low dimensional domains. 
Training with transformed _(x, y)_ coordinates shows dramatic improvements in the results and can preserve the sharp edges in the image. 


## References

1. [NeRF Paper](https://arxiv.org/abs/2003.08934)
2. [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)
3. [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)
4. [NeRF Video](https://www.youtube.com/watch?v=nRyOzHpcr4Q&t=1706s)

## License

MIT

---

### Image Credits
- [Unsplash](https://unsplash.com)
- [POV-Ray Hall of Fame](https://hof.povray.org/)