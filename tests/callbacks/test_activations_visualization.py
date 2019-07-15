import os
import shutil
from pathlib import Path

import tensorflow as tf

from mentat.callbacks.activations_visualization import ActivationsVisualizationCallback


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
        ActivationsVisualizationCallback(
            validation_data=(x, None),
            layers_name=['activation_1'],
            output_dir=output_dir
        )
    ]

    model.fit(x, y, batch_size=2, epochs=2, callbacks=callbacks)

    assert len(os.listdir(Path(output_dir))) == 2

    shutil.rmtree(output_dir)
