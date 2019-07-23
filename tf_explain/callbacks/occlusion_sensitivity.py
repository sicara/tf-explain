from pathlib import Path

from tensorflow.keras.callbacks import Callback

from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity


class OcclusionSensitivityCallback(Callback):
    def __init__(
        self,
        validation_data,
        patch_size,
        class_index,
        output_dir=Path("./logs/occlusion_sensitivity"),
    ):
        super(OcclusionSensitivityCallback, self).__init__()
        self.validation_data = validation_data
        self.patch_size = patch_size
        self.class_index = class_index
        self.output_dir = output_dir
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        explainer = OcclusionSensitivity()
        grid = explainer.explain(
            self.validation_data, self.model, self.class_index, self.patch_size
        )
        explainer.save(grid, self.output_dir, f"{epoch}.png")
