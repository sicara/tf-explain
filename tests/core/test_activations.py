import numpy as np

from tf_explain.core.activations import ExtractActivations


def test_should_generate_subgraph(convolutional_model):
    activations_model = ExtractActivations.generate_activations_graph(
        convolutional_model, ["activation_1"]
    )

    assert activations_model.layers[-1].name == "activation_1"


def test_should_extract_activations(random_data, convolutional_model, mocker):
    non_normalized_grid = np.array([[1, 2], [1, 2]])
    mocker.patch(
        "tf_explain.core.activations.filter_display", return_value=non_normalized_grid
    )
    explainer = ExtractActivations()
    grid = explainer.explain(random_data, convolutional_model, ["activation_1"])

    expected_output = np.array([[0, 255], [0, 255]]).astype("uint8")

    np.testing.assert_array_equal(grid, expected_output)


def test_should_save_output_grid(output_dir):
    grid = np.random.random((208, 208))

    explainer = ExtractActivations()
    explainer.save(grid, output_dir, "output.png")

    assert len(list(output_dir.glob("output.png"))) == 1
