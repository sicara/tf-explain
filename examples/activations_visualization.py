from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tf_explain.utils.display import filter_display

layers_name = ['activation_6']
IMAGE_PATH = './cat.jpg'

if __name__ == '__main__':
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model.summary()
    outputs = [
        layer.output for layer in model.layers
        if layer.name in layers_name
    ]

    activations_model = tf.keras.models.Model(model.inputs, outputs=outputs)
    activations_model.compile(optimizer='adam', loss='categorical_crossentropy')

    predictions = activations_model.predict(np.array([img]))
    grid = filter_display(predictions)

    im = Image.fromarray((np.clip(grid, 0, 1) * 255).astype('uint8'))
    im.save(Path('../tests/test_logs') / f'1.png')
