import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, backend
from tensorflow.keras.preprocessing import image
from evaluation import evaluate_model, specificity
import matplotlib.pyplot as plt
import numpy as np
#import tensorflowjs as tfjs

datagen_train = image.ImageDataGenerator(rescale=1./255)
datagen_test = image.ImageDataGenerator(rescale=1./255)

folder_path = './KDEF'
train_set = datagen_train.flow_from_directory(folder_path+"/", color_mode='grayscale', target_size=(224, 224),
                                              class_mode='categorical')

test_set = datagen_test.flow_from_directory(folder_path+"/", color_mode='grayscale', target_size=(224, 224),
                                            class_mode='categorical')


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
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

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', specificity])

model.summary()
epochs = 60

model.fit(train_set, validation_data=test_set, epochs=epochs)

print("Classification report with train data set")
evaluate_model(model, train_set)
print("Classification report with test data set")
evaluate_model(model, test_set)

y_pred = model.predict(train_set)
y_pred = np.argmax(y_pred, axis=1)
class_labels = train_set.class_indices
print(y_pred)
print(class_labels)

tfjs.converters.save_keras_model(model, './')


# https://www.kaggle.com/code/songdevelop/facial-emotion-recogination
# https://www.kaggle.com/code/amankumar2004/emotion-detection-recognition
# https://towardsdatascience.com/deploying-an-image-classifier-using-javascript-84da1480b3a4
# https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
