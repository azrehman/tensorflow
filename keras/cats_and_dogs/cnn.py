import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D)

X = np.load('X.npy')
y = np.load('y.npy')
