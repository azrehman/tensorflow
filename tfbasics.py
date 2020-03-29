import numpy as np
import os
os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf



def tf_no_warning():
    """
    Make Tensorflow less verbose
    """
    try:

        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    except ImportError:
        pass
tf_no_warning()

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# metaparams
n_hidden_nodes = [500, 500, 500]
n_classes = 10
bach_size = 100

#                          shape for input
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    # Layer =  X.W + b
    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([784, n_hidden_nodes[0]])),
        'biases': tf.Variable(tf.random_normal([n_hidden_nodes[0]]))
        }
    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([n_hidden_nodes[0], n_hidden_nodes[1]])),
        'biases': tf.Variable(tf.random_normal([n_hidden_nodes[1]]))
        }
    hidden_3_layer = {
        'weights': tf.Variable(tf.random_normal([n_hidden_nodes[1], n_hidden_nodes[2]])),
        'biases': tf.Variable(tf.random_normal([n_hidden_nodes[2]]))
        }
        
    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_hidden_nodes[2], n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
        }


neural_network_model(None)
print(0)