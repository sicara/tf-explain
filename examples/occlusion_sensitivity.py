import tensorflow as tf

from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

IMAGE_PATH = './cat.jpg'

if __name__ == '__main__':
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model.summary()
    data = ([img], None)

    tabby_cat_class_index = 281
    explainer = OcclusionSensitivity()
    # Compute Occlusion Sensitivity for patch_size 20
    grid = explainer.explain(data, model, tabby_cat_class_index, 20)
    explainer.save(grid, '.', 'occlusion_sensitivity_20.png')
    # Compute Occlusion Sensitivity for patch_size 10
    grid = explainer.explain(data, model, tabby_cat_class_index, 10)
    explainer.save(grid, '.', 'occlusion_sensitivity_10.png')
