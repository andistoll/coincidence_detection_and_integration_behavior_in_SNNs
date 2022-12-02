import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
import numpy as np


brightness_offset_neg = 40
# THIS OFFSET IS SUBTRACTED

brightness_offset_pos = 10.4 #10
# THIS OFFSET IS ADDED


def load_local_emnist(path, dataset='letters', kind='train'):
    import os
    import gzip

    """Load eMNIST data from `path`"""
    labels_path = os.path.join(path, f'emnist-{dataset}-{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'emnist-{dataset}-{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def load_letter_mnist_with_negative_brightness_offset(num_folds=5):
    # load and split the dataset into train and test sets
    data_path = "/home/hpc/iwi5/iwi5084h/coincidence_detection_in_snns/training/emnist"
    x_train_set, y_train_set = load_local_emnist(data_path, dataset='letters', kind='train')
    x_test, y_test = load_local_emnist(data_path, dataset='letters', kind='test')
    
    train_fold = []
    val_fold = []

    x_train_set = deepcopy(x_train_set)
    x_test = deepcopy(x_test)
    x_train_set = np.asarray(x_train_set, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    
    x_test[x_test <= brightness_offset_neg] = 0
    x_test[x_test > brightness_offset_neg] -= brightness_offset_neg
    

    skf = StratifiedKFold(n_splits=num_folds, random_state=7, shuffle=True)
    for train_index, val_index in skf.split(np.zeros(y_train_set.size),y_train_set):
            x_train = x_train_set[train_index]
            y_train = y_train_set[train_index]

            x_val = x_train_set[val_index]
            y_val = y_train_set[val_index]

            
            x_train[x_train <= brightness_offset_neg] = 0
            x_train[x_train > brightness_offset_neg] -= brightness_offset_neg
            
            x_val[x_val <= brightness_offset_neg] = 0
            x_val[x_val > brightness_offset_neg] -= brightness_offset_neg

            train_fold.append((x_train, y_train))
            val_fold.append((x_val, y_val))

    return train_fold, val_fold, (x_test, y_test)


def load_letter_mnist_with_negative_brightness_offset_for_evaluation():
    path = f"./training/emnist"

    x_train, y_train = load_local_emnist(path, dataset='letters', kind='train')
    x_test, y_test = load_local_emnist(path, dataset='letters', kind='test')
    x_train = deepcopy(x_train)
    x_test = deepcopy(x_test)
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    
    x_test[x_test <= brightness_offset_neg] = 0
    x_test[x_test > brightness_offset_neg] -= brightness_offset_neg
    x_train[x_train <= brightness_offset_neg] = 0
    x_train[x_train > brightness_offset_neg] -= brightness_offset_neg


    return (x_train, y_train), (x_test, y_test)


def load_letter_mnist_with_positive_brightness_offset(num_folds=5):
    # load and split the dataset into train and test sets
    data_path = "/home/hpc/iwi5/iwi5084h/coincidence_detection_in_snns/training/emnist"
    x_train_set, y_train_set = load_local_emnist(data_path, dataset='letters', kind='train')
    x_test, y_test = load_local_emnist(data_path, dataset='letters', kind='test')

    x_train_set = deepcopy(x_train_set)
    x_test = deepcopy(x_test)
    x_train_set = np.asarray(x_train_set, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    
    train_fold = []
    val_fold = []

    x_test[x_test > 255 - brightness_offset_pos] = 255
    x_test[x_test <= 255 - brightness_offset_pos] += brightness_offset_pos
    

    skf = StratifiedKFold(n_splits=num_folds, random_state=7, shuffle=True)
    for train_index, val_index in skf.split(np.zeros(y_train_set.size),y_train_set):
            x_train = x_train_set[train_index]
            y_train = y_train_set[train_index]

            x_val = x_train_set[val_index]
            y_val = y_train_set[val_index]

            x_train[x_train > 255 - brightness_offset_pos] = 255
            x_train[x_train <= 255 - brightness_offset_pos] += brightness_offset_pos
            x_val[x_val > 255 - brightness_offset_pos] = 255
            x_val[x_val <= 255 - brightness_offset_pos] += brightness_offset_pos

            train_fold.append((x_train, y_train))
            val_fold.append((x_val, y_val))

    return train_fold, val_fold, (x_test, y_test)


def load_letter_mnist_with_positive_brightness_offset_for_evaluation():
    path = f"./training/emnist"

    x_train, y_train = load_local_emnist(path, dataset='letters', kind='train')
    x_test, y_test = load_local_emnist(path, dataset='letters', kind='test')

    x_train = deepcopy(x_train)
    x_test = deepcopy(x_test)
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    
    x_test[x_test > 255 - brightness_offset_pos] = 255
    x_test[x_test <= 255 - brightness_offset_pos] += brightness_offset_pos

    x_train[x_train > 255 - brightness_offset_pos] = 255
    x_train[x_train <= 255 - brightness_offset_pos] += brightness_offset_pos


    return (x_train, y_train), (x_test, y_test)
