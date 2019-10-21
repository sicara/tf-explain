import numpy as np

from tf_explain.callbacks.vanilla_gradients import VanillaGradientsCallback


def test_should_call_vanilla_gradients_callback(
    random_data, convolutional_model, output_dir, mocker
):
    images, labels = random_data

    vanilla_gradient_callback = VanillaGradientsCallback(
        validation_data=random_data, class_index=0, output_dir=output_dir
    )

    mock_explainer = mocker.MagicMock(
        explain=mocker.MagicMock(return_value=np.zeros((28, 28)))
    )

    vanilla_gradient_callback.explainer = mock_explainer

    callbacks = [vanilla_gradient_callback]

    convolutional_model.fit(images, labels, batch_size=2, epochs=1, callbacks=callbacks)

    mock_explainer.explain.assert_called_once_with(random_data, convolutional_model, 0)
    assert len(list(output_dir.iterdir())) == 1
