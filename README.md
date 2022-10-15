<h1 align="center">
  <br>
  nerf_toy
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

|                                                Original                                                |                                                    Raw                                                     | Basic Encoding                                                                                               |                                              Fourier Feature                                               |
|:------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|--------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------:|
| <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/data/lion_face.jpg" width="175"> | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/raw_lion_face.gif" width="175"> | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/basic_lion_face.gif" width="175"> | <img src="https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/misc/rff_lion_face.gif" width="175"> |

</div>

## Description

**nerf-toy** is a 2D toy illustration of the [Neural Radiance Fields](http://www.matthewtancik.com/nerf). Memorizing a
2D image using an MLP is a nice problem to understand before going into solving the problem
NeRF tackles i.e. memorising a 3D scene given multiple view points.

The task is to produce the _(r, g, b)_ value of an image as a
function of _(x,y)_ image coordinates. Our model is a coordinate-based multilayer perceptron.

![Network Diagram](https://raw.githubusercontent.com/shubhamwagh/nerf-toy/main/data/network_diagram.png)

This is implemented using 2D convolutions with a kernel size of (1, 1), which act as "**pointwise**" convolutions. This
is equivalent to a densely connected multilayer perceptron for each coordinate.
Also, we use batchnorm to speed up convergence.

[//]: # (Script to add LaTeX maths equation)
<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>

We compare the following mappings $\gamma (\mathbf{v})$.

- No mapping: $\gamma(\mathbf{v})= \mathbf{v}$. 

- Basic mapping: $\gamma(\mathbf{v})=\left[ \cos(2 \pi \mathbf{v}),\sin(2 \pi \mathbf{v}) \right]^\mathrm{T}$. 

<!-- - Positional encoding: $\gamma(\mathbf{v})=\left[ \ldots, \cos(2 \pi \sigma^{j/m} \mathbf{v}),\sin(2 \pi \sigma^{j/m} \mathbf{v}), \ldots \right]^\mathrm{T}$ for $j = 0, \ldots, m-1$.  -->

- Gaussian Fourier feature mapping: $\gamma(\mathbf{v})= \left[ \cos(2 \pi \mathbf B \mathbf{v}), \sin(2 \pi \mathbf B \mathbf{v}) \right]^\mathrm{T}$, 
where each entry in $\mathbf B \in \mathbb R^{m \times d}$ is sampled from $\mathcal N(0,\sigma^2)$

## Refrences

1. NeRF - https://arxiv.org/abs/2003.08934
2. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains - https://arxiv.org/abs/2006.10739
3. Neural Tangent Kernel: Convergence and Generalization in Neural Networks - https://arxiv.org/abs/1806.07572
