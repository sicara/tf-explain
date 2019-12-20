import numpy as np
import pytest
import tensorflow as tf
import tf_explain


INPUT_SHAPE = (28, 28, 1)


def functional_api_model(num_classes):
    img_input = tf.keras.Input(INPUT_SHAPE)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(
        img_input
    )
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation="relu", name="grad_cam_target"
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)

    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(img_input, x)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def sequential_api_model(num_classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=INPUT_SHAPE,
            ),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                name="grad_cam_target",
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def subclassing_api_model(num_classes):
    class SubclassedModel(tf.keras.models.Model):
        def __init__(self, name="subclassed"):
            super(SubclassedModel, self).__init__(name=name)
            self.conv_1 = tf.keras.layers.Conv2D(
                filters=32, kernel_size=(3, 3), activation="relu"
            )
            self.conv_2 = tf.keras.layers.Conv2D(
                filters=64, kernel_size=(3, 3), activation="relu"
            )
            self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

            self.conv_3 = tf.keras.layers.Conv2D(
                filters=32, kernel_size=(3, 3), activation="relu"
            )
            self.conv_4 = tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                name="grad_cam_target",
            )
            self.maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

            self.flatten = tf.keras.layers.Flatten()

            self.dense_1 = tf.keras.layers.Dense(64, activation="relu")

            self.dense_2 = tf.keras.layers.Dense(num_classes, activation="softmax")

        def build(self, input_shape):
            super(SubclassedModel, self).build(input_shape)

        def call(self, inputs, **kwargs):
            x = inputs
            for layer in [
                self.conv_1,
                self.conv_2,
                self.maxpool_1,
                self.conv_3,
                self.conv_4,
                self.maxpool_2,
                self.flatten,
                self.dense_1,
                self.dense_2,
            ]:
                x = layer(x)

            return x

        def compute_output_shape(self, input_shape):
            shape = tf.TensorShape(input_shape).as_list()
            return tf.TensorShape([shape[0], num_classes])

    model = SubclassedModel()

    model(
        np.random.random([4, *INPUT_SHAPE]).astype("float32")
    )  # Sample call to build the model

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# TODO: Activate subclassing API model (issue #55)
@pytest.mark.parametrize("model_builder", [functional_api_model, sequential_api_model])
def test_all_keras_api(
    model_builder, mnist_dataset, validation_dataset, num_classes, output_dir
):
    train_images, train_labels, test_images, test_labels = mnist_dataset

    model = model_builder(num_classes)
    model.summary()

    validation_data, target_class = validation_dataset

    # Instantiate callbacks
    callbacks = [
        tf_explain.callbacks.GradCAMCallback(
            validation_data,
            layer_name="grad_cam_target",
            class_index=target_class,
            output_dir=output_dir,
        ),
        tf_explain.callbacks.ActivationsVisualizationCallback(
            validation_data, "grad_cam_target", output_dir=output_dir
        ),
        tf_explain.callbacks.SmoothGradCallback(
            validation_data,
            class_index=target_class,
            num_samples=15,
            noise=1.0,
            output_dir=output_dir,
        ),
        tf_explain.callbacks.VanillaGradientsCallback(
            validation_data, class_index=target_class, output_dir=output_dir
        ),
        tf_explain.callbacks.GradientsInputsCallback(
            validation_data, class_index=target_class, output_dir=output_dir
        ),
    ]

    # Start training
    model.fit(train_images, train_labels, epochs=3, callbacks=callbacks)
