import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D)
from tensorflow.keras.callbacks import TensorBoard

# set up keras callback
MODEL_NAME = 'Cats-vs-Dogs-CNN-64x2-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(MODEL_NAME))


# load dataset generated from parse_images.py
X = np.load('X.npy')
y = np.load('y.npy')

# normalize input
X = X / 255.0

print(X.shape[1:])
# model architecture
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# convert 3D feature maps to 1D feature vectors
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))

# output layer
model.add(Dense(1))
model.add(Activation('softmax'))

# train model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=4, epochs=5,
          validation_split=0.3, callbacks=[tensorboard])
