import shutil

import numpy as np
from tf_explain.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback


def test_should_call_occlusion_sensitivity_callback(
    random_data, convolutional_model, output_dir, mocker
):
    mock_explainer = mocker.MagicMock()
    mock_explainer.explain = mocker.MagicMock(return_value=0)
    mocker.patch(
        "tf_explain.callbacks.occlusion_sensitivity.OcclusionSensitivity",
        return_value=mock_explainer,
    )
    mock_image_summary = mocker.patch(
        "tf_explain.callbacks.occlusion_sensitivity.tf.summary.image"
    )

    images, labels = random_data

    callbacks = [
        OcclusionSensitivityCallback(
            validation_data=random_data,
            class_index=0,
            patch_size=10,
            output_dir=output_dir,
        )
    ]

    convolutional_model.fit(images, labels, batch_size=2, epochs=1, callbacks=callbacks)

    mock_explainer.explain.assert_called_once_with(
        random_data, convolutional_model, 0, 10
    )
    mock_image_summary.assert_called_once_with(
        "Occlusion Sensitivity", np.array([0]), step=0
    )

    shutil.rmtree(output_dir)
