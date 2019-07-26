import shutil

import numpy as np
from tf_explain.callbacks.activations_visualization import (
    ActivationsVisualizationCallback,
)


def test_should_call_activations_visualization_callback(
    random_data, convolutional_model, output_dir, mocker
):
    mock_explainer = mocker.MagicMock()
    mock_explainer.explain = mocker.MagicMock(return_value=0)
    mocker.patch(
        "tf_explain.callbacks.activations_visualization.ExtractActivations",
        return_value=mock_explainer,
    )
    mock_image_summary = mocker.patch(
        "tf_explain.callbacks.occlusion_sensitivity.tf.summary.image"
    )

    images, labels = random_data

    callbacks = [
        ActivationsVisualizationCallback(
            validation_data=random_data,
            layers_name=["activation_1"],
            output_dir=output_dir,
        )
    ]

    convolutional_model.fit(images, labels, batch_size=2, epochs=1, callbacks=callbacks)

    mock_explainer.explain.assert_called_once_with(
        random_data, convolutional_model, ["activation_1"]
    )
    mock_image_summary.assert_called_once_with(
        "Activations Visualization", np.array([[0]]), step=0
    )
    shutil.rmtree(output_dir)
