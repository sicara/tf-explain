import numpy as np
from tf_explain.callbacks.smoothgrad import SmoothGradCallback


def test_should_call_smoothgrad_callback(
    random_data, convolutional_model, output_dir, mocker
):
    mock_explainer = mocker.MagicMock(
        explain=mocker.MagicMock(return_value=np.zeros((28, 28)))
    )
    mocker.patch(
        "tf_explain.callbacks.smoothgrad.SmoothGrad", return_value=mock_explainer
    )

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
    assert len([_ for _ in output_dir.iterdir()]) == 1
