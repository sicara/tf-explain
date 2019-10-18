import numpy as np
import tensorflow as tf

from tf_explain.core.gradients_inputs import GradientsInputs


IMAGE_PATH = "examples/core/cat.jpg"

if __name__ == "__main__":
    model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model.summary()
    data = (np.array([img]), None)

    tabby_cat_class_index = 281
    explainer = GradientsInputs()
    # Compute GradientsInputs on VGG16
    grid = explainer.explain(data, model, tabby_cat_class_index)
    explainer.save(grid, ".", "gradients_inputs.png")
