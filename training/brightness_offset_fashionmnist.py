import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold

brightness_offset = 42 #90
# THIS OFFSET IS SUBTRACTED


def load_local_fashion_mnist(path, kind='train'):
    import os
    import gzip

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def load_fashion_mnist_with_brightness_offset(num_folds=5):
    # load and split the dataset into train and test sets
    #(x_train_set, y_train_set), test_set = keras.datasets.fashion_mnist.load_data()
    data_path = "/home/hpc/iwi5/iwi5084h/coincidence_detection_in_snns/training/fashion_mnist"
    x_train_set, y_train_set = load_local_fashion_mnist(data_path, kind='train')
    x_test, y_test = load_local_fashion_mnist(data_path, kind='t10k')
    train_fold = []
    val_fold = []

    x_train_set = deepcopy(x_train_set)
    x_test = deepcopy(x_test)
    
    x_test[x_test <= brightness_offset] = 0
    x_test[x_test > brightness_offset] -= brightness_offset
    

    skf = StratifiedKFold(n_splits=num_folds, random_state=7, shuffle=True)
    for train_index, val_index in skf.split(np.zeros(y_train_set.size),y_train_set):
            x_train = x_train_set[train_index]
            y_train = y_train_set[train_index]

            x_val = x_train_set[val_index]
            y_val = y_train_set[val_index]

            
            x_train[x_train <= brightness_offset] = 0
            x_train[x_train > brightness_offset] -= brightness_offset
            
            x_val[x_val <= brightness_offset] = 0
            x_val[x_val > brightness_offset] -= brightness_offset

            train_fold.append((x_train, y_train))
            val_fold.append((x_val, y_val))

    return train_fold, val_fold, (x_test, y_test)


def load_fashion_mnist_with_brightness_offset_for_evaluation():
    # load and split the dataset into train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    x_train = deepcopy(x_train)
    x_test = deepcopy(x_test)
    
    x_test[x_test <= brightness_offset] = 0
    x_test[x_test > brightness_offset] -= brightness_offset
    x_train[x_train <= brightness_offset] = 0
    x_train[x_train > brightness_offset] -= brightness_offset

    return (x_train, y_train), (x_test, y_test)
