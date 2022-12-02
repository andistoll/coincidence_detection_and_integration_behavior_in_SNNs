"""
This script shows the training pipeline for k-fold training, without logging it via wandb.
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

from sklearn.model_selection import StratifiedKFold
from tf_spiking.helper import TrainingHistory
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
import numpy as np



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


with tf.device('GPU:0'):
    print("EMNIST/Letters")
    decay_initializers = ['constant']
    decays = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]  # [ms]

    #decay_initializers = ['binned_uniform']
    #decays = [15]  # [ms]

    
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

            train_fold, val_fold, test_set = load_letter_mnist(num_folds)

            for fold in range(0, num_folds):

                (x_train, y_train) = train_fold[fold]
                (x_val, y_val) = val_fold[fold]
                (x_test, y_test) = test_set

                # generate the model
                num_lif_units = 128
                num_classes = 27

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
                model_name = f"letter_mnist_time{time_steps}_dt{dt}_poisson_hidden{num_lif_units}_{decay_distribution}_decay{t_dec}ms_fold{fold}"
                model.save(f"./models/dt{dt}_untrained/{experiment}/" + model_name)


                # fit the model
                output_path = f"letter_mnist_time{time_steps}_dt{dt}_poisson_hidden{num_lif_units}"
                model.fit(x_train, keras.utils.to_categorical(y_train), validation_data=(x_val, keras.utils.to_categorical(y_val)),
                        batch_size=256, epochs=30, verbose=0, # verbose=0 == silent
                        callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max'),
                                    keras.callbacks.ModelCheckpoint(output_path+"/weights.hdf5", save_best_only=True, save_weights_only=True),
                                    TrainingHistory(output_path),
                                ]
                )

                # Save the trained model
                model_name = f"letter_mnist_time{time_steps}_dt{dt}_poisson_hidden{num_lif_units}_{decay_distribution}_decay{t_dec}ms_fold{fold}"
                model.save(f"./models/dt{dt}/{experiment}/" + model_name)

                test_loss, test_accuracy, test_auc, test_f1 = model.evaluate(x_test, keras.utils.to_categorical(y_test), verbose=0)
                #print("*" * 40)
                print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test AUC: {test_auc}, Test F1: {test_f1}")
