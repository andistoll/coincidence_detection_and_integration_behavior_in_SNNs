import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import numpy as np

from sklearn.model_selection import StratifiedKFold


brightness_offset = 22 #43
# THIS OFFSET IS ADDED

def load_mnist_with_brightness_offset(num_folds=5):
    # load and split the dataset into train and test sets
    (x_train_set, y_train_set), (x_test, y_test) = keras.datasets.mnist.load_data(path="/home/hpc/iwi5/iwi5084h/coincidence_detection_in_snns/training/mnist.npz")
    train_fold = []
    val_fold = []
  
    x_test[x_test > 255 - brightness_offset] = 255
    x_test[x_test <= 255 - brightness_offset] += brightness_offset
    

    skf = StratifiedKFold(n_splits=num_folds, random_state=7, shuffle=True)
    for train_index, val_index in skf.split(np.zeros(y_train_set.size),y_train_set):
            x_train = x_train_set[train_index]
            y_train = y_train_set[train_index]

            x_val = x_train_set[val_index]
            y_val = y_train_set[val_index]

            x_train[x_train > 255 - brightness_offset] = 255
            x_train[x_train <= 255 - brightness_offset] += brightness_offset
            x_val[x_val > 255 - brightness_offset] = 255
            x_val[x_val <= 255 - brightness_offset] += brightness_offset

            train_fold.append((x_train, y_train))
            val_fold.append((x_val, y_val))

    return train_fold, val_fold, (x_test, y_test)


def load_mnist_with_brightness_offset_for_evaluation():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  
    x_test[x_test > 255 - brightness_offset] = 255
    x_test[x_test <= 255 - brightness_offset] += brightness_offset

    x_train[x_train > 255 - brightness_offset] = 255
    x_train[x_train <= 255 - brightness_offset] += brightness_offset

    return (x_train, y_train), (x_test, y_test)
