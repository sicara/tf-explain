import numpy as np

from tf_explain.callbacks.vanilla_gradients import VanillaGradientsCallback


def test_should_call_vanilla_gradients_callback(
    random_data, convolutional_model, output_dir, mocker
):
    images, labels = random_data
    score_model = convolutional_model

    vanilla_gradient_callback = VanillaGradientsCallback(
        validation_data=random_data, class_index=0, output_dir=output_dir
    )

    mock_explainer = mocker.MagicMock(
        get_score_model=mocker.MagicMock(return_value=score_model),
        explain_score_model=mocker.MagicMock(return_value=np.zeros((28, 28))),
    )

    vanilla_gradient_callback.explainer = mock_explainer

    callbacks = [vanilla_gradient_callback]

    convolutional_model.fit(images, labels, batch_size=2, epochs=1, callbacks=callbacks)

    mock_explainer.get_score_model.assert_called_once()
    mock_explainer.explain_score_model.assert_called_once_with(
        random_data, score_model, 0
    )
    assert len(list(output_dir.iterdir())) == 1


def test_should_only_compute_score_model_once(
    random_data, convolutional_model, output_dir, mocker
):
    """
    Tests that the Vanilla Gradients explainer only computes the score model once
    during training. This improves performance as it ensures the gradients are
    always calculated with the same score model, which prevents tf.function retracing.
    """

    images, labels = random_data
    score_model = convolutional_model

    vanilla_gradient_callback = VanillaGradientsCallback(
        validation_data=random_data, class_index=0, output_dir=output_dir
    )

    mock_explainer = mocker.MagicMock(
        get_score_model=mocker.MagicMock(return_value=score_model),
        explain_score_model=mocker.MagicMock(return_value=np.zeros((28, 28))),
    )

    vanilla_gradient_callback.explainer = mock_explainer

    callbacks = [vanilla_gradient_callback]

    # Two epochs
    convolutional_model.fit(images, labels, batch_size=2, epochs=2, callbacks=callbacks)

    # Score model only computed once
    mock_explainer.get_score_model.assert_called_once()
