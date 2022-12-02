import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import StratifiedKFold


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


def load_letter_mnist(num_folds=5):
    # load and split the dataset into train and test sets
    data_path = "/home/hpc/iwi5/iwi5084h/coincidence_detection_in_snns/training/emnist"
    x_train_set, y_train_set = load_local_emnist(data_path, dataset='letters', kind='train')
    test_set = load_local_emnist(data_path, dataset='letters', kind='test')
    
    
    train_fold = []
    val_fold = []

    skf = StratifiedKFold(n_splits=num_folds, random_state=7, shuffle=True)
    for train_index, val_index in skf.split(np.zeros(y_train_set.size),y_train_set):
            x_train = x_train_set[train_index]
            y_train = y_train_set[train_index]

            x_val = x_train_set[val_index]
            y_val = y_train_set[val_index]

            train_fold.append((x_train, y_train))
            val_fold.append((x_val, y_val))

    return train_fold, val_fold, test_set


def load_local_letter_mnist_for_evaluation():
    path = f"./training/emnist"

    x_train, y_train = load_local_emnist(path, dataset='letters', kind='train')
    x_test, y_test = load_local_emnist(path, dataset='letters', kind='test')

    return (x_train, y_train), (x_test, y_test)



images, labels = load_local_emnist("./training/emnist")
print(np.unique(labels))
