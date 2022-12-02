"""
This script shows the training pipeline for k-fold training, without logging it via wandb.
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from tf_spiking.helper import TrainingHistory
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
import numpy as np
import sys
import pickle
import os


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = '/home/hpc/iwi5/iwi5084h/coincidence_detection_in_snns/training/cifar-10-batches-py'

    num_train_samples = 50000

    x_train_local = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train_local = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train_local[(i - 1) * 10000: i * 10000, :, :, :],
         y_train_local[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test_local, y_test_local = load_batch(fpath)

    y_train_local = np.reshape(y_train_local, (len(y_train_local), 1))
    y_test_local = np.reshape(y_test_local, (len(y_test_local), 1))

    if K.image_data_format() == 'channels_last':
        x_train_local = x_train_local.transpose(0, 2, 3, 1)
        x_test_local = x_test_local.transpose(0, 2, 3, 1)

    return (x_train_local, y_train_local), (x_test_local, y_test_local)


def load_cifar10(num_folds=5):
    # load and split the dataset into train and test sets
    #(x_train_set, y_train_set), test_set = keras.datasets.cifar10.load_data()
    (x_train_set, y_train_set), test_set = load_data()

    train_fold = []
    val_fold = []

    skf = StratifiedKFold(n_splits=num_folds, random_state=7, shuffle=True)
    for train_index, val_index in skf.split(np.zeros(y_train_set.size), y_train_set):
            x_train = x_train_set[train_index]
            y_train = y_train_set[train_index]

            x_val = x_train_set[val_index]
            y_val = y_train_set[val_index]

            train_fold.append((x_train, y_train))
            val_fold.append((x_val, y_val))

    return train_fold, val_fold, test_set


with tf.device('GPU:0'):
    print("CIFAR-10")
    decay_initializers = ['constant']
    decays = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]  # [ms]

    #decay_initializers = ['binned_uniform']
    #decays = [15]
    
    
    gaussian_weights = False
    only_pos_weights=False
    use_bias = True

    experiment = "no_bias_gaussian_weights_128" # TODO: make sure this folder exists in dt{dt} and dt{dt}_untrained

    if experiment.__contains__("no_bias"):
        use_bias = False
    if experiment.__contains__("gaussian_weights"):
        gaussian_weights = True

    for decay_distribution in decay_initializers:
        print(f"current decay distribution: {decay_distribution}")

        for t_dec in decays:

            num_folds = 5

            train_fold, val_fold, test_set = load_cifar10(num_folds)

            for fold in range(0, num_folds):

                (x_train, y_train) = train_fold[fold]
                (x_val, y_val) = val_fold[fold]
                (x_test, y_test) = test_set

                # generate the model
                num_lif_units = 128
                num_classes = 10

                simulation_time = 0.5    # duration of stimuli   [s]
                dt = 0.005               # temportal resolution  [s]
                time_steps = int(simulation_time / dt)
                t_decay = t_dec / 1000 # decay time of Vm      [ms]

                model = keras.models.Sequential([
                    keras.Input(x_train.shape[1:]),
                    IntensityToPoissonSpiking(time_steps, 1/255, dt=dt),
                    DenseLIF(
                        num_lif_units,
                        gaussian_weights=gaussian_weights,
                        only_pos_weights=only_pos_weights,
                        use_bias=use_bias, 
                        surrogate="flat", 
                        dt=dt,
                        decay_distribution=decay_distribution,
                        trainable_leak=False, 
                        t_decay=t_decay),
                    DenseLIFNoSpike(num_classes, dt),
                ])

                #print(model.summary())
                print(f"Starting fold no. {fold} with t_decay={t_dec}ms ...")

                # Configure training
                optim = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
                loss = keras.losses.MeanSquaredError()
                metrics = [ 'accuracy',
                            'AUC',
                            tfa.metrics.F1Score(num_classes=num_classes, average='macro')]

                # compile the model
                model.compile(
                    optimizer=optim,
                    loss=loss,
                    metrics=metrics,
                )

                # Save untrained model for sanity checks
                model_name = f"cifar10_time{time_steps}_dt{dt}_poisson_hidden{num_lif_units}_{decay_distribution}_decay{t_dec}ms_fold{fold}"
                model.save(f"./trained_models/dt{dt}_untrained/{experiment}/" + model_name)


                # fit the model
                output_path = f"cifar10_time{time_steps}_dt{dt}_poisson_hidden{num_lif_units}"
                model.fit(x_train, keras.utils.to_categorical(y_train), validation_data=(x_val, keras.utils.to_categorical(y_val)),
                        batch_size=256, epochs=30, verbose=0, # verbose=0 == silent
                        callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max'),
                                    keras.callbacks.ModelCheckpoint(output_path+"/weights.hdf5", save_best_only=True, save_weights_only=True),
                                    TrainingHistory(output_path),
                                ]
                )

                # Save the trained model
                model_name = f"cifar10_time{time_steps}_dt{dt}_poisson_hidden{num_lif_units}_{decay_distribution}_decay{t_dec}ms_fold{fold}"
                model.save(f"./trained_models/dt{dt}/{experiment}/" + model_name)

                test_loss, test_accuracy, test_auc, test_f1 = model.evaluate(x_test, keras.utils.to_categorical(y_test), verbose=0)
                #print("*" * 40)
                print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test AUC: {test_auc}, Test F1: {test_f1}")
