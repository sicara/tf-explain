"""
Callback Module for Gradients*Inputs algorithm
"""
from tf_explain.callbacks.vanilla_gradients import VanillaGradientsCallback
from tf_explain.core import GradientsInputs


class GradientsInputsCallback(VanillaGradientsCallback):

    """
    Tensorflow Callback performing Gradients*Inputs algorithm for given input and target class
    """

    explainer = GradientsInputs()
    default_output_subdir = "gradients_inputs"
