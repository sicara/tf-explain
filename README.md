# Visualize

[![Build Status](https://api.travis-ci.com/sicara/mentat.svg?branch=master)](https://travis-ci.org/sicara/mentat)
![Python Versions](https://img.shields.io/badge/python-3.6%20|%203.7-%23EBBD68.svg)
![Tensorflow Versions](https://img.shields.io/badge/tensorflow-2.0.0--beta1-blue.svg)

Interpretability of Deep Learning Models with Tensorflow 2.0

## Examples

Here is a list of all available callbacks. All those examples are generated with the `examples` scripts.

### Activations Visualization

> Visualize how a given input comes out of a specific activation layer

<p align="center">
    <img src="./docs/assets/activations_visualisation.png" width="500" />
</p>


### Occlusion Sensitivity

> Visualize how parts of the image affects neural network's confidence by occluding parts iteratively

<p align="center">
    <img src="./docs/assets/occlusion_sensitivity.png" width="200" />
</p>

### Grad CAM

> Visualize how parts of the image affects neural network's output by looking into the activation maps

From [Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

<p align="center">
    <img src="./docs/assets/grad_cam.png" width="200" />
</p>


## Roadmap

Next features are listed as issues with the `roadmap` label.
