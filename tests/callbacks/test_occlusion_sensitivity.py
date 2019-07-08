import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from visualize.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback


def test_should_call_activations_visualization_callback(random_data):
    x, y = random_data
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation=None, name='conv_1', input_shape=list(x.shape[1:])),
        tf.keras.layers.ReLU(name='activation_1'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    output_dir = os.path.join('tests', 'test_logs')
    callbacks = [
        OcclusionSensitivityCallback(
            validation_data=(x, None),
            patch_size=4,
            class_index=0,
            output_dir=output_dir,
        )
    ]

    model.fit(x, y, batch_size=2, epochs=2, callbacks=callbacks)

    assert len(os.listdir(Path(output_dir))) == 2

    shutil.rmtree(output_dir)


def test_should_get_sensitivity_map(random_data, mocker):
    mocker.patch('visualize.callbacks.occlusion_sensitivity.OcclusionSensitivityCallback.get_confidence_for_random_patch', return_value=0.6)

    x, y = random_data

    output_dir = os.path.join('tests', 'test_logs')
    callback = OcclusionSensitivityCallback(
            validation_data=(x, None),
            patch_size=4,
            class_index=0,
            output_dir=output_dir,
    )
    # callback.get_confidence_for_random_patch = mocker.MagicMock(return_value=0.6)
    output = callback.get_sensitivity_map(
        model=None,
        image=x[0],
        class_index=None,
        patch_size=4,
    )

    expected_output = 0.4 * np.ones((x[0].shape[0], x[0].shape[1]))

    np.testing.assert_almost_equal(output, expected_output)
