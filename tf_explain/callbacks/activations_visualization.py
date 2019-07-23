from pathlib import Path

from tensorflow.keras.callbacks import Callback

from tf_explain.core.activations import ExtractActivations


class ActivationsVisualizationCallback(Callback):

    """ Draw activations of a specific layer for a given input """

    def __init__(
        self,
        validation_data,
        layers_name,
        output_dir=Path("./logs/activations_visualizations"),
    ):
        super(ActivationsVisualizationCallback, self).__init__()
        self.validation_data = validation_data
        self.layers_name = layers_name
        self.output_dir = output_dir
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """ Draw activations outputs at each epoch end. """
        explainer = ExtractActivations()
        grid = explainer.explain(self.validation_data, self.model, self.layers_name)
        explainer.save(grid, self.output_dir, f"{epoch}.png")
