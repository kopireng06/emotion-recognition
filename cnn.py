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

folder_path = './KDEF'
train_set = datagen_train.flow_from_directory(folder_path+"/", color_mode='grayscale', target_size=(224, 224), batch_size=64,
                                              class_mode='categorical')

test_set = datagen_test.flow_from_directory(folder_path+"/", color_mode='grayscale', target_size=(224, 224),batch_size=64,
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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
epochs = 20

model.fit(train_set, validation_data=test_set, epochs=epochs)

tfjs.converters.save_keras_model(model, './')

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

test_ds = tf.keras.utils.image_dataset_from_directory(
  './KDEF',
  image_size=(224, 224),
  batch_size=64,
  color_mode="grayscale"
)

evaluate_model(model, test_ds, class_labels)


# https://stackoverflow.com/questions/62556931/huge-difference-between-in-accuracy-between-model-evaluate-and-model-predict-for
# https://www.kaggle.com/code/o1anuraganand/emotion-model-vgg16-transfer-learning-training
# https://www.kaggle.com/code/songdevelop/facial-emotion-recogination
# https://www.kaggle.com/code/amankumar2004/emotion-detection-recognition
# https://towardsdatascience.com/deploying-an-image-classifier-using-javascript-84da1480b3a4
# https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
