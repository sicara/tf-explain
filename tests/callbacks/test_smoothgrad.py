import shutil

import numpy as np
from tf_explain.callbacks.smoothgrad import SmoothGradCallback


def test_should_call_grad_cam_callback(
    random_data, convolutional_model, output_dir, mocker
):
    mock_explainer = mocker.MagicMock()
    mock_explainer.explain = mocker.MagicMock(return_value=0)
    mocker.patch(
        "tf_explain.callbacks.smoothgrad.SmoothGrad", return_value=mock_explainer
    )
    mock_image_summary = mocker.patch("tf_explain.callbacks.grad_cam.tf.summary.image")

    images, labels = random_data

    callbacks = [
        SmoothGradCallback(
            validation_data=random_data,
            class_index=0,
            output_dir=output_dir,
            num_samples=3,
            noise=1.2,
        )
    ]

    convolutional_model.fit(images, labels, batch_size=2, epochs=1, callbacks=callbacks)

    mock_explainer.explain.assert_called_once_with(
        random_data, convolutional_model, 0, 3, 1.2
    )
    mock_image_summary.assert_called_once_with("SmoothGrad", np.array([0]), step=0)

    shutil.rmtree(output_dir)
