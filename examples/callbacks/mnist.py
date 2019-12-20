import numpy as np
import tensorflow as tf
import tf_explain

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

AVAILABLE_DATASETS = {
    "mnist": tf.keras.datasets.mnist,
    "fashion_mnist": tf.keras.datasets.fashion_mnist,
}
DATASET_NAME = "fashion_mnist"  # Choose between "mnist" and "fashion_mnist"

# Load dataset
dataset = AVAILABLE_DATASETS[DATASET_NAME]
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

# Convert from (28, 28) images to (28, 28, 1)
train_images = train_images[..., tf.newaxis].astype("float32") / 255.0
test_images = test_images[..., tf.newaxis].astype("float32") / 255.0

# One hot encore labels 0, 1, .., 9 to [0, 0, .., 1, 0, 0]
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)

# Create model
img_input = tf.keras.Input(INPUT_SHAPE)

x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(img_input)
x = tf.keras.layers.Conv2D(
    filters=64, kernel_size=(3, 3), activation="relu", name="target_layer"
)(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(img_input, x)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Select a subset of the validation data to examine
# Here, we choose 5 elements with one hot encoded label "0" == [1, 0, 0, .., 0]
validation_class_zero = (
    np.array(
        [
            el
            for el, label in zip(test_images, test_labels)
            if np.all(np.argmax(label) == 0)
        ][0:5]
    ),
    None,
)
# Select a subset of the validation data to examine
# Here, we choose 5 elements with one hot encoded label "4" == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
validation_class_four = (
    np.array(
        [
            el
            for el, label in zip(test_images, test_labels)
            if np.all(np.argmax(label) == 4)
        ][0:5]
    ),
    None,
)

# Instantiate callbacks
# class_index value should match the validation_data selected above
callbacks = [
    tf_explain.callbacks.GradCAMCallback(
        validation_class_zero, class_index=0, layer_name="target_layer"
    ),
    tf_explain.callbacks.GradCAMCallback(
        validation_class_four, class_index=4, layer_name="target_layer"
    ),
    tf_explain.callbacks.ActivationsVisualizationCallback(
        validation_class_zero, layers_name=["target_layer"]
    ),
    tf_explain.callbacks.SmoothGradCallback(
        validation_class_zero, class_index=0, num_samples=15, noise=1.0
    ),
    tf_explain.callbacks.IntegratedGradientsCallback(
        validation_class_zero, class_index=0, n_steps=10
    ),
    tf_explain.callbacks.VanillaGradientsCallback(validation_class_zero, class_index=0),
    tf_explain.callbacks.GradientsInputsCallback(validation_class_zero, class_index=0),
]

# Start training
model.fit(train_images, train_labels, epochs=5, callbacks=callbacks)
