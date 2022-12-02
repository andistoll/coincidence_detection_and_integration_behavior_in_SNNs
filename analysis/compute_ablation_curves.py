from copy import deepcopy
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory
from training.brightness_offset_mnist import load_mnist_with_brightness_offset_for_evaluation
from training.brightness_offset_fashionmnist import load_fashion_mnist_with_brightness_offset_for_evaluation
from training.letter_mnist import load_local_letter_mnist_for_evaluation


dataset = "mnist" # options: mnist, letter_mnist, fashion_mnist, cifar10 -- mnist_with_brightness_offset, fashion_mnist_with_brightness_offset
experiment = "no_bias_gaussian_weights_128" # options: no_bias_gaussian_weights_128, no_bias_gaussian_weights_128_adj


intervals = 16 # specifies into how many intervals decay_times is split and set to 0 respectively
cumulative_dropout = True

for i in range(2):
    if i == 0:
        descending_decay = True
    else:
        descending_decay = False


    x_ticks = np.arange(0, intervals + 1, 1)
    x_ticks = np.round((((x_ticks - 0.0) * 100) / (intervals+ 1)), 0)
    x_ticks = list(map(int, x_ticks))

    # constant does not make sense here!!!
    initializer_distributions = ['binned_uniform']
    t_decay = 15

    # Default configuration - will be automatically reconfigured according to the name of the experiment
    num_folds = 5
    gaussian_weights = False
    use_bias = True
    simulation_time = 0.5
    dt = 0.005
    folder_name = f"dt{dt}"
    
    num_neurons = 128
    decay_range = [15, 480]

    # If this parameter changes you will get an error; the default use_bias in Dense_LIF (integrate_and_fire.py) has to be changed by hand!
    if experiment.__contains__("no_bias"):
        use_bias = False

    if experiment.__contains__("gaussian_weights"):
        gaussian_weights = True


    timesteps = int(simulation_time / dt)

    # load the dataset
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        max_acc = 0.9
    elif dataset == "mnist_with_brightness_offset":
        (x_train, y_train), (x_test, y_test) = load_mnist_with_brightness_offset_for_evaluation()
        max_acc = 0.8
    elif dataset == "letter_mnist":
        (x_train, y_train), (x_test, y_test) = load_local_letter_mnist_for_evaluation()
        max_acc = 0.7
    elif dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        max_acc = 0.8
    elif dataset == "fashion_mnist_with_brightness_offset":
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist_with_brightness_offset_for_evaluation()
        max_acc = 0.8
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        max_acc = 0.4
    else:
        raise NotImplementedError(f"{dataset} needs to be loaded...")


    for distrib in initializer_distributions:

        # dicts are per initializer
        mean_accs = {}
        std_accs = {}

        from_percent = 0.0
        to_percent = 0.0

        for interval in range(intervals + 1):

            accs = []

            for fold in range(num_folds):

                # Setup model and load weights
                # generate the model
                num_lif_units = num_neurons
                if dataset == "letter_mnist":
                    num_classes = 27
                else:
                    num_classes = 10

                t_dec = t_decay / 1000 # decay time of Vm      [ms]

                model = keras.models.Sequential([
                    keras.Input(x_test.shape[1:]),
                    IntensityToPoissonSpiking(timesteps, 1/255, dt=dt),
                    DenseLIF(
                        num_lif_units,
                        gaussian_weights=gaussian_weights,
                        use_bias=use_bias,
                        surrogate="flat", 
                        dt=dt,
                        decay_distribution=distrib,
                        trainable_leak=False, 
                        t_decay=t_dec),
                    DenseLIFNoSpike(num_classes, dt),
                ])
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

                path = f"./models/{folder_name}/{experiment}/{dataset}_time{timesteps}_dt{dt}_poisson_hidden{num_neurons}_{distrib}_decay{t_decay}ms_fold{fold}/"
                saved_model = keras.models.load_model(path)

                model.set_weights(saved_model.get_weights())


                if use_bias:
                    [w_input, bias, w_leak] = saved_model.layers[1].get_weights()
                else:
                    [w_input, w_leak] = saved_model.layers[1].get_weights()

                decay_times = 1 / w_leak
                decay_total_size = decay_times.size

                if descending_decay:
                    decay_argsort = np.argsort(decay_times)[::-1]
                else:
                    decay_argsort = np.argsort(decay_times)


                neurons = decay_argsort[int(from_percent * decay_total_size) : int(to_percent * decay_total_size)]

                w_input[:, neurons] = 0


                if use_bias:
                    model.layers[1].set_weights([w_input, bias, w_leak])
                else:
                    model.layers[1].set_weights([w_input, w_leak])
                
                # Recompute the performance metrics
                test_loss, test_accuracy, test_auc, test_f1 = model.evaluate(x_test, keras.utils.to_categorical(y_test), verbose=0)
                    
                accs.append(deepcopy(test_accuracy))

                keras.backend.clear_session()

            mean_accs[f'{distrib}_from_{int(from_percent * 100)}_to_{int(to_percent * 100)}'] = np.mean(accs)
            std_accs[f'{distrib}_from_{int(from_percent * 100)}_to_{int(to_percent * 100)}'] = np.std(accs)

            if not cumulative_dropout:
                from_percent = to_percent
            to_percent += 1 / intervals


        if os.path.exists(f'./results/{dataset}/{experiment}/{distrib}_intervals{intervals}.csv'):
                os.remove(f'./results/{dataset}/{experiment}/{distrib}_intervals{intervals}.csv')

        dict = {
            "mean_acc": mean_accs,
            "std_acc": std_accs,
        }

        df = pd.DataFrame.from_dict(dict)
        df.to_csv(f'./results/{dataset}/{experiment}/{distrib}_intervals{intervals}_descending={descending_decay}.csv')

