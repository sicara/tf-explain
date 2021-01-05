Available Methods
#################

Activations Visualization
*************************

Visualize how a given input comes out of a specific activation layer
::
    from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback

    model = [...]

    callbacks = [
        ActivationsVisualizationCallback(
            validation_data=(x_val, y_val),
            layers_name=["activation_1"],
            output_dir=output_dir,
        ),
    ]

    model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)

.. image:: ../assets/activations_visualisation.png
   :alt: Activations Visualization
   :width: 400px
   :align: center


Vanilla Gradients
*****************

Visualize gradients on the inputs towards the decision.

From `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps <https://arxiv.org/abs/1312.6034)>`_
::
    from tf_explain.callbacks.vanilla_gradients import VanillaGradientsCallback

    model = [...]
    callbacks = [
        VanillaGradientsCallback(
            validation_data=(x_val, y_val),
            class_index=0,
            output_dir=output_dir,
        )
    ]

    model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)

.. image:: ../assets/vanilla_gradients.png
   :alt: VanillaGradients
   :width: 200px
   :align: center


Gradients*Inputs
*****************

Variant of Vanilla Gradients ponderating gradients with input values.
::
    from tf_explain.callbacks.gradients_inputs import GradientsInputsCallback

    model = [...]

    callbacks = [
        GradientsInputsCallback(
            validation_data=(x_val, y_val),
            class_index=0,
            output_dir=output_dir,
        ),
    ]

    model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)

.. image:: ../assets/gradients_inputs.png
   :alt: GradientsInputs
   :width: 200px
   :align: center


Occlusion Sensitivity
*********************

Visualize how parts of the image affects neural network's confidence by occluding parts iteratively
::
    from tf_explain.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback

    model = [...]

    callbacks = [
        OcclusionSensitivityCallback(
            validation_data=(x_val, y_val),
            class_index=0,
            patch_size=4,
            output_dir=output_dir,
        ),
    ]

    model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)

.. image:: ../assets/occlusion_sensitivity.png
   :alt: Occlusion Sensitivity
   :width: 200px
   :align: center


Grad CAM
********

Visualize how parts of the image affects neural network's output by looking into the activation maps

From `Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization <https://arxiv.org/abs/1610.02391>`_
::
    from tf_explain.callbacks.grad_cam import GradCAMCallback

    model = [...]

    callbacks = [
        GradCAMCallback(
            validation_data=(x_val, y_val),
            layer_name="activation_1",
            class_index=0,
            output_dir=output_dir,
        )
    ]

    model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)

.. image:: ../assets/grad_cam.png
   :alt: GradCAM
   :width: 200px
   :align: center


SmoothGrad
**********

Visualize stabilized gradients on the inputs towards the decision.

From `SmoothGrad: removing noise by adding noise <https://arxiv.org/abs/1706.03825>`_
::
    from tf_explain.callbacks.smoothgrad import SmoothGradCallback

    model = [...]

    callbacks = [
        SmoothGradCallback(
            validation_data=(x_val, y_val),
            class_index=0,
            num_samples=20,
            noise=1.,
            output_dir=output_dir,
        )
    ]

    model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)

.. image:: ../assets/smoothgrad.png
   :alt: SmoothGrad
   :width: 200px
   :align: center


Integrated Gradients
********************

Visualize an average of the gradients along the construction of the input towards the decision.

From `Axiomatic Attribution for Deep Networks <https://arxiv.org/pdf/1703.01365.pdf>`_
::
    from tf_explain.callbacks.integrated_gradients import IntegratedGradientsCallback

    model = [...]

    callbacks = [
        IntegratedGradientsCallback(
            validation_data=(x_val, y_val),
            class_index=0,
            n_steps=20,
            output_dir=output_dir,
        )
    ]

    model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)

.. image:: ../assets/integrated_gradients.png
   :alt: IntegratedGradients
   :width: 200px
   :align: center
