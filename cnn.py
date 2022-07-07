import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

train_images = keras.preprocessing.image_dataset_from_directory(
    './FER2013/train', image_size=(48, 48), color_mode='grayscale')
test_images = keras.preprocessing.image_dataset_from_directory(
    './FER2013/test', image_size=(48, 48), color_mode='grayscale')


datagen_train = image.ImageDataGenerator(rescale=1./255, shear_range=0.2)
datagen_test = image.ImageDataGenerator(rescale=1./255, shear_range=0.2)

folder_path = './FER2013'
train_set = datagen_train.flow_from_directory(folder_path+"/train", color_mode='grayscale',
                                              class_mode='categorical', shuffle=True)

test_set = datagen_test.flow_from_directory(folder_path+"/test", color_mode='grayscale',
                                            class_mode='categorical', shuffle=True)

# for data, labels in dataset:
#   print(data.shape)  # (64, 200, 200, 3)
#   print(data.dtype)  # float32
#   print(labels.shape)  # (64,)
#   print(labels.dtype)  # int32

# inputs = keras.Input(shape=(48, 48, 3))

# x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)


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

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()
epochs = 60

history4 = model.fit(train_images, validation_data=test_images, epochs=epochs)

# Define Sequential model with 3 layers
# model = keras.Sequential(
#     [
#         layers.Dense(2, activation="relu", name="layer1"),
#         layers.Dense(3, activation="relu", name="layer2"),
#         layers.Dense(4, name="layer3"),
#     ]
# )
# Call model on a test input
# x = tf.ones((3, 3))
# y = model(x)
