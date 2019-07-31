# tf-explain

[![Pypi Version](https://img.shields.io/pypi/v/tf-explain.svg)](https://pypi.org/project/tf-explain/)
[![Build Status](https://api.travis-ci.org/sicara/tf-explain.svg?branch=master)](https://travis-ci.org/sicara/tf-explain)
[![Documentation Status](https://readthedocs.org/projects/tf-explain/badge/?version=latest)](https://tf-explain.readthedocs.io/en/latest/?badge=latest)
![Python Versions](https://img.shields.io/badge/python-3.6%20|%203.7-%23EBBD68.svg)
![Tensorflow Versions](https://img.shields.io/badge/tensorflow-2.0.0--beta1-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

__tf-explain__ implements interpretability methods as Tensorflow 2.0 callbacks to __ease neural network's understanding__.  
See [Introducing tf-explain, Interpretability for Tensorflow 2.0](https://blog.sicara.com/tf-explain-interpretability-tensorflow-2-9438b5846e35)

## Installation

__tf-explain__ is available on PyPi as an alpha release. To install it:

```bash
virtualenv venv -p python3.6
pip install tf-explain
```

tf-explain is compatible with Tensorflow 2. It is not declared as a dependency
to let you choose between CPU and GPU versions. Additionally to the previous install, run:

```bash
# For CPU version
pip install tensorflow==2.0.0-beta1
# For GPU version
pip install tensorflow-gpu==2.0.0-beta1
```

## Available Methods

1. [Activations Visualization](#activations-visualization)
2. [Occlusion Sensitivity](#occlusion-sensitivity)
3. [Grad CAM (Class Activation Maps)](#grad-cam)
4. [SmoothGrad](#smoothgrad)

### Activations Visualization

> Visualize how a given input comes out of a specific activation layer

```python
from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback

model = [...]

callbacks = [
    ActivationsVisualizationCallback(
        validation_data=(x_val, y_val),
        layers_name=["activation_1"],
        output_dir=output_dir,
    ),
]

model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)
```

<p align="center">
    <img src="./docs/assets/activations_visualisation.png" width="400" />
</p>


### Occlusion Sensitivity

> Visualize how parts of the image affects neural network's confidence by occluding parts iteratively

```python
from tf_explain.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback

model = [...]

callbacks = [
    OcclusionSensitivityCallback(
        validation_data=(x_val, y_val),
        class_index=0,
        patch_size=4,
        output_dir=output_dir,
    ),
]

model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)
```

<div align="center">
    <img src="./docs/assets/occlusion_sensitivity.png" width="200" />
    <p style="color: grey; font-size:small; width:350px;">Occlusion Sensitivity for Tabby class (stripes differentiate tabby cat from other ImageNet cat classes)</p>
</div>

### Grad CAM

> Visualize how parts of the image affects neural network's output by looking into the activation maps

From [Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

```python
from tf_explain.callbacks.grad_cam import GradCAMCallback

model = [...]

callbacks = [
    GradCAMCallback(
        validation_data=(x_val, y_val),
        layer_name="activation_1",
        class_index=0,
        output_dir=output_dir,
    )
]

model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)
```


<p align="center">
    <img src="./docs/assets/grad_cam.png" width="200" />
</p>

### SmoothGrad

> Visualize stabilized gradients on the inputs towards the decision

From [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)

```python
from tf_explain.callbacks.smoothgrad import SmoothGradCallback

model = [...]

callbacks = [
    SmoothGradCallback(
        validation_data=(x_val, y_val),
        class_index=0,
        num_samples=20,
        noise=1.,
        output_dir=output_dir,
    )
]

model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)
```

<p align="center">
    <img src="./docs/assets/smoothgrad.png" width="200" />
</p>



## Visualizing the results

When you use the callbacks, the output files are created in the `logs` directory.

You can see them in tensorboard with the following command: `tensorboard --logdir logs`


## Roadmap

- [ ] Subclassing API Support
- [ ] Additional Methods
  - [ ] [GradCAM++](https://arxiv.org/abs/1710.11063)
  - [ ] [Integrated Gradients](https://arxiv.org/abs/1703.01365)
  - [ ] [Guided SmoothGrad](https://arxiv.org/abs/1706.03825)
  - [ ] [LRP](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
- [ ] Auto-generated API Documentation & Documentation Testing
