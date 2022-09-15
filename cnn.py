import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, backend
from tensorflow.keras.preprocessing import image
from evaluation import evaluate_model, specificity
import matplotlib.pyplot as plt
import numpy as np
import tensorflowjs as tfjs

datagen_train = image.ImageDataGenerator(rescale=1./255)
datagen_test = image.ImageDataGenerator(rescale=1./255)

class_labels = ['angry', 'disgusted',' fear', 'happy', 'neutral', 'sad', 'surprised']

train_set = tf.keras.utils.image_dataset_from_directory(
  './FER2013_ORIGINAL/train',
  label_mode='categorical',
  image_size=(48, 48),
  batch_size=64,
  color_mode="grayscale"
)

test_set = tf.keras.utils.image_dataset_from_directory(
  './FER2013_ORIGINAL/test',
  label_mode='categorical',
  image_size=(48, 48),
  batch_size=64,
  color_mode="grayscale"
)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Flatten())
model.add(layers.Dense(750, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(850, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(750, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation="softmax"))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
epochs = 60

model.fit(train_set, validation_data=test_set, epochs=epochs)

tfjs.converters.save_keras_model(model, './')

# serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

test = tf.keras.utils.image_dataset_from_directory(
  './FER2013_ORIGINAL/test',
  image_size=(48, 48),
  batch_size=64,
  color_mode="grayscale"
)

evaluate_model(model, test, class_labels)


# https://stackoverflow.com/questions/62556931/huge-difference-between-in-accuracy-between-model-evaluate-and-model-predict-for
# https://www.kaggle.com/code/o1anuraganand/emotion-model-vgg16-transfer-learning-training
# https://www.kaggle.com/code/songdevelop/facial-emotion-recogination
# https://www.kaggle.com/code/amankumar2004/emotion-detection-recognition
# https://towardsdatascience.com/deploying-an-image-classifier-using-javascript-84da1480b3a4
# https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
