from pathlib import Path

import cv2
import tensorflow as tf

from mentat.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback


IMAGE_PATH = './cat.jpg'

if __name__ == '__main__':
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model.summary()

    tabby_cat_class_index = 281
    sensitivity_map = OcclusionSensitivityCallback.get_sensitivity_map(model, img, tabby_cat_class_index, 10)

    sensitivity_jet_map = cv2.applyColorMap(
        cv2.cvtColor((sensitivity_map * 255).astype('uint8'), cv2.COLOR_GRAY2RGB),
        cv2.COLORMAP_JET,
    )

    heatmap_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 1, sensitivity_jet_map, 0.5, 0)
    cv2.imwrite(str(Path('../tests/test_logs') / 'occlusion.png'), heatmap_image)
