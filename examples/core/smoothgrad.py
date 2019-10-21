import tensorflow as tf

from tf_explain.core.smoothgrad import SmoothGrad

IMAGE_PATH = "./cat.jpg"

if __name__ == "__main__":
    model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model.summary()
    data = ([img], None)

    tabby_cat_class_index = 281
    explainer = SmoothGrad()
    # Compute SmoothGrad on VGG16
    grid = explainer.explain(data, model, tabby_cat_class_index, 20, 1.0)
    explainer.save(grid, ".", "smoothgrad.png")
