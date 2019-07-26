import numpy as np
import tensorflow as tf

from tf_explain.core.activations import ExtractActivations

layers_name = ['activation_6']
IMAGE_PATH = './cat.jpg'

if __name__ == '__main__':
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model.summary()
    data = (np.array([img]), None)

    explainer = ExtractActivations()
    # Compute Activations of layer activation_1
    grid = explainer.explain(data, model, ['activation_6'])
    explainer.save(grid, '.', 'activations.png')
