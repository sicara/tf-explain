
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


Occlusion Sensitivity
*********************

Visualize how parts of the image affects neural network's confidence by occluding parts iteratively
::
    from tf_explain.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback

    model = [...]

    callbacks = [
        OcclusionSensitivityCallback(
            validation_data=(x_val, y_val),
            patch_size=4,
            class_index=0,
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
