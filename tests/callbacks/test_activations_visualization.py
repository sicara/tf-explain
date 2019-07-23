import shutil
from pathlib import Path

from tf_explain.callbacks.activations_visualization import (
    ActivationsVisualizationCallback,
)


def test_should_call_activations_visualization_callback(
    random_data, convolutional_model
):
    x, y = random_data

    output_dir = Path("tests") / "test_logs"
    callbacks = [
        ActivationsVisualizationCallback(
            validation_data=(x, None),
            layers_name=["activation_1"],
            output_dir=output_dir,
        )
    ]

    convolutional_model.fit(x, y, batch_size=2, epochs=2, callbacks=callbacks)

    assert len(list(output_dir.glob("*"))) == 2

    shutil.rmtree(output_dir)
