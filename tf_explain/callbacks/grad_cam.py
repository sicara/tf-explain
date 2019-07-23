import os
from pathlib import Path

from tensorflow.keras.callbacks import Callback

from tf_explain.core.grad_cam import GradCAM


class GradCAMCallback(Callback):

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def __init__(
        self,
        validation_data,
        layer_name,
        class_index,
        output_dir=Path("./logs/grad_cam"),
    ):
        super(GradCAMCallback, self).__init__()
        self.validation_data = validation_data
        self.layer_name = layer_name
        self.class_index = class_index
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """ Draw activations outputs at each epoch end. """
        explainer = GradCAM()
        heatmap = explainer.explain(
            self.validation_data, self.model, self.layer_name, self.class_index
        )
        explainer.save(heatmap, self.output_dir, f"{epoch}.png")
