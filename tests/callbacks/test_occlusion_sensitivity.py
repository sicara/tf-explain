import os
import shutil
from pathlib import Path

import numpy as np

from mentat.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback


def test_should_call_occlusion_sensitivity_callback(random_data, convolutional_model):
    x, y = random_data
    convolutional_model.compile(optimizer='adam', loss='categorical_crossentropy')

    output_dir = os.path.join('tests', 'test_logs')
    callbacks = [
        OcclusionSensitivityCallback(
            validation_data=(x, None),
            patch_size=4,
            class_index=0,
            output_dir=output_dir,
        )
    ]

    convolutional_model.fit(x, y, batch_size=2, epochs=2, callbacks=callbacks)

    assert len(os.listdir(Path(output_dir))) == 2

    shutil.rmtree(output_dir)


def test_should_get_sensitivity_map(random_data, mocker):
    mocker.patch('mentat.callbacks.occlusion_sensitivity.OcclusionSensitivityCallback.get_confidence_for_random_patch', return_value=0.6)

    x, y = random_data

    output_dir = os.path.join('tests', 'test_logs')
    callback = OcclusionSensitivityCallback(
            validation_data=(x, None),
            patch_size=4,
            class_index=0,
            output_dir=output_dir,
    )

    output = callback.get_sensitivity_map(
        model=None,
        image=x[0],
        class_index=None,
        patch_size=4,
    )

    expected_output = 0.4 * np.ones((x[0].shape[0], x[0].shape[1]))

    np.testing.assert_almost_equal(output, expected_output)
