import tensorflow as tf
from tensorflow import keras
import os

# zoom_range=0.2,

generatorImage = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            shear_range=0.2,
                                                            rotation_range=0.3,
                                                            horizontal_flip=True
                                                            )

dir_path = './disgust_for_augmentation'

count = 0
for root, dirs, files in os.walk(dir_path):
    for filename in files:
        count += 1

imageSource = keras.preprocessing.image_dataset_from_directory(dir_path, color_mode='rgb', 
                                                               batch_size=count, image_size=(48, 48))

totalGeneratePerImage = 6 # determine how many generated per image
length_dataset = list(imageSource.unbatch().as_numpy_iterator())


def generateImage(image, totalGenerate):
    i = 0
    for batch in generatorImage.flow(image, batch_size=1, save_to_dir='./disgust_generated', save_prefix='dgst', save_format='jpeg'):
        i += 1
        if i >= totalGenerate:
            break
        
for images, labels in imageSource.take(1):
    for x in range(len(length_dataset)):
        img = tf.keras.preprocessing.image.img_to_array(images[x])
        img = img.reshape((1,) + img.shape)
        generateImage(img, totalGeneratePerImage)
        
