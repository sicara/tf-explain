from tf_explain.core.gradients_inputs import GradientsInputs


def test_get_ponderated_gradients(random_data, convolutional_model):
    images, _ = random_data
    gradients = GradientsInputs.compute_gradients(images, convolutional_model, 0)

    assert gradients.shape == images.shape
