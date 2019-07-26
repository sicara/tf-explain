Usage
#####

tf-explain implements methods you can use at different levels:

* either on a loaded model with the core API (which saves outputs to disk)
* either at training time with callbacks (which integrates into Tensorboard)

This section introduces both usages.


Core API
********

All methods implemented in tf-explain keep the same interface:

* a :code:`explain` method which outputs the explaination (for instance, a heatmap)
* a :code:`save` method compatible with its output

Usage of the core API should be the following:
::
    # Import explainer
    from tf_explain.core.grad_cam import GradCAM

    # Instantiation of the explainer
    explainer = GradCAM()

    # Call to explain() method
    output = explainer.explain(*explainer_args)

    # Save output
    explainer.save(output, output_dir, output_name)

Recurrent arguments contained in :code:`explainer_args` are typically the data to use
for the explanation, the model to inspect. Refer to each method docstring to know which
elements are needed.

All methods are kept inside :code:`tf_explain.core`.

Callbacks
*********

To use those methods during trainings and inspect evolutions over the epochs, each one of them
has its corresponding :code:`tf.keras.Callback`.

Callback usage is coherent with Keras Callbacks:
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

Then, launch `Tensorboard <https://www.tensorflow.org/tensorboard/>`_ and visualize the outputs in the Images section.
