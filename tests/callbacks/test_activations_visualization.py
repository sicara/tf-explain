import os
import shutil
from pathlib import Path

from mentat.callbacks.activations_visualization import ActivationsVisualizationCallback


def test_should_call_activations_visualization_callback(random_data, convolutional_model):
    x, y = random_data

    output_dir = os.path.join('tests', 'test_logs')
    callbacks = [
        ActivationsVisualizationCallback(
            validation_data=(x, None),
            layers_name=['activation_1'],
            output_dir=output_dir
        )
    ]

    convolutional_model.fit(x, y, batch_size=2, epochs=2, callbacks=callbacks)

    assert len(os.listdir(Path(output_dir))) == 2

    shutil.rmtree(output_dir)
