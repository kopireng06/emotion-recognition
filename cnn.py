import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
from confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

train_images = keras.preprocessing.image_dataset_from_directory(
    './FER2013/train', image_size=(48, 48), color_mode='grayscale')
test_images = keras.preprocessing.image_dataset_from_directory(
    './FER2013/test', image_size=(48, 48), color_mode='grayscale')


datagen_train = image.ImageDataGenerator(rescale=1./255)
datagen_test = image.ImageDataGenerator(rescale=1./255)

folder_path = './FER2013'
train_set = datagen_train.flow_from_directory(folder_path+"/train", color_mode='grayscale', target_size=(48, 48), 
                                              class_mode='categorical', shuffle=True)

test_set = datagen_test.flow_from_directory(folder_path+"/test", color_mode='grayscale', target_size=(48, 48), 
                                            class_mode='categorical', shuffle=True)


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

history4 = model.fit(train_set, validation_data=test_set, epochs=epochs)

y_pred = model.predict(train_set)
print("y_pred")
y_pred = np.argmax(y_pred, axis=1)
class_labels = train_set.class_indices
class_labels = {v:k for k,v in class_labels.items()}

from sklearn.metrics import classification_report, confusion_matrix
cm_train = confusion_matrix(train_set.classes, y_pred)
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(train_set.classes, y_pred, target_names=target_names))
plot_confusion_matrix(cm_train, target_names)

y_pred = model.predict(test_set)
print("y_pred")
y_pred = np.argmax(y_pred, axis=1)
class_labels = test_set.class_indices
class_labels = {v:k for k,v in class_labels.items()}

from sklearn.metrics import classification_report, confusion_matrix
cm_test = confusion_matrix(test_set.classes, y_pred)
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(test_set.classes, y_pred, target_names=target_names))
plot_confusion_matrix(cm_test, target_names)

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, './')


# https://www.kaggle.com/code/songdevelop/facial-emotion-recogination
# https://www.kaggle.com/code/amankumar2004/emotion-detection-recognition
# https://towardsdatascience.com/deploying-an-image-classifier-using-javascript-84da1480b3a4