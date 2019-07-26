import shutil

import numpy as np
from tf_explain.callbacks.grad_cam import GradCAMCallback


def test_should_call_grad_cam_callback(
    random_data, convolutional_model, output_dir, mocker
):
    mock_explainer = mocker.MagicMock()
    mock_explainer.explain = mocker.MagicMock(return_value=0)
    mocker.patch("tf_explain.callbacks.grad_cam.GradCAM", return_value=mock_explainer)
    mock_image_summary = mocker.patch("tf_explain.callbacks.grad_cam.tf.summary.image")

    images, labels = random_data

    callbacks = [
        GradCAMCallback(
            validation_data=random_data,
            class_index=0,
            layer_name=["activation_1"],
            output_dir=output_dir,
        )
    ]

    convolutional_model.fit(images, labels, batch_size=2, epochs=1, callbacks=callbacks)

    mock_explainer.explain.assert_called_once_with(
        random_data, convolutional_model, ["activation_1"], 0
    )
    mock_image_summary.assert_called_once_with("Grad CAM", np.array([0]), step=0)

    shutil.rmtree(output_dir)
