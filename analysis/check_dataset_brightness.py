from copy import deepcopy
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

from sklearn.model_selection import StratifiedKFold
from tf_spiking.helper import TrainingHistory
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
import numpy as np
from training.letter_mnist import load_local_letter_mnist_for_evaluation


# MNIST
brightness_offset = 22
# load and split the dataset into train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

mnist_brightness = (np.sum(x_train) + np.sum(x_test)) / (x_train.shape[0] + x_test.shape[0])


print("MNIST: ", mnist_brightness / (28*28))

x_test[x_test > 255 - brightness_offset] = 255
x_test[x_test <= 255 - brightness_offset] += brightness_offset
x_train[x_train > 255 - brightness_offset] = 255
x_train[x_train <= 255 - brightness_offset] += brightness_offset

m_b = (np.sum(x_train) + np.sum(x_test)) / (x_train.shape[0] + x_test.shape[0])
print("MNIST adj.: ", m_b / (28 * 28))


# lettersMNIST
(x_train, y_train), (x_test, y_test) = load_local_letter_mnist_for_evaluation()
x_train = np.asarray(x_train, dtype=float)
x_test = np.asarray(x_test, dtype=float)
lettermnist_brightness = (np.sum(x_train) + np.sum(x_test)) / (x_train.shape[0] + x_test.shape[0])
print("EMNIST/Letters: ", lettermnist_brightness / (28*28))
brightness_offset = 10.4
x_train = deepcopy(x_train)
x_test = deepcopy(x_test)
x_test[x_test > 255 - brightness_offset] = 255
x_test[x_test <= 255 - brightness_offset] += brightness_offset
x_train[x_train > 255 - brightness_offset] = 255
x_train[x_train <= 255 - brightness_offset] += brightness_offset

l_b = (np.sum(x_train) + np.sum(x_test)) / (x_train.shape[0] + x_test.shape[0])
print("EMNIST/Letters adj: ",l_b / (28*28))


# fashionMNIST
# load and split the dataset into train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
fashionmnist_brightness = (np.sum(x_train) + np.sum(x_test)) / (x_train.shape[0] + x_test.shape[0])
print("Fashion-MNIST: ", fashionmnist_brightness / (28*28))
brightness_offset = 42
x_train = deepcopy(x_train)
x_test = deepcopy(x_test)
x_test[x_test <= brightness_offset] = 0
x_test[x_test > brightness_offset] -= brightness_offset
x_train[x_train <= brightness_offset] = 0
x_train[x_train > brightness_offset] -= brightness_offset

f_b = (np.sum(x_train) + np.sum(x_test)) / (x_train.shape[0] + x_test.shape[0])
print("Fashion-MNIST adj.: ",f_b / (28*28))



print("max difference of adj. data: ", np.max( [np.abs(m_b / (28*28) - f_b / (28*28)), np.abs(l_b / (28*28) - f_b / (28*28)), np.abs(m_b / (28*28) - l_b / (28*28))]))

# cifar10
# load and split the dataset into train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

cifar10_brightness = (np.sum(x_train) + np.sum(x_test)) / (x_train.shape[0] + x_test.shape[0])
print("CIFAR-10: ", cifar10_brightness / (32*32*3))
