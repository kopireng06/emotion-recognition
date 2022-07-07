import tensorflow as tf
from tensorflow import keras

putin_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            shear_range=0.2,
                                                            rotation_range=0.3,
                                                            zoom_range=0.2,
                                                            horizontal_flip=True)

putin_image = keras.preprocessing.image_dataset_from_directory(
    './augmentation-test', color_mode='rgb', batch_size=436, image_size=(48, 48))

length_dataset = list(putin_image.unbatch().as_numpy_iterator())
print(len(length_dataset))

dataset = tf.data.Dataset.range(10)
print(dataset.take(3))


def generateImage(a):
    i = 0
    for batch in putin_gen.flow(img, batch_size=1,
                                save_to_dir='./disgust_generated', save_prefix='disgust', save_format='jpeg'):
        i += 1
        if i >= 10:
            break


for images, labels in putin_image.take(1):
    for x in range(len(length_dataset)):
        print(x)
        img = tf.keras.preprocessing.image.img_to_array(images[x])
        img = img.reshape((1,) + img.shape)
        print(labels[0])
        generateImage(img)
