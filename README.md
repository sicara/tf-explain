# Visualize

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

- [ ] Improve performance for callbacks (occlusion, gradcam)
- [ ] Add other method (SmoothGrad for instance)
- [ ] Make all heatmap visualizations centralized and uniform
