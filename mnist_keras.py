# mnist with tf and and keras based on:
# https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/


import tensorflow as tf

# retrieve and load data
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# model architecture
model = tf.keras.models.Sequential()
# flatten 28x28 image data to 1x784 arrays
model.add(tf.keras.layers.Flatten())
# 2 hidden layers with 128 neurons using ReLU activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# output layer with 10 neurons using softmax activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model parameters
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrirrcs=['accuracy'])

# train model
model.fit(X_train, y_train, epochs=3)

# evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(loss)
print(acc)

# to save model:
# model.save('mnist_keras.model')
