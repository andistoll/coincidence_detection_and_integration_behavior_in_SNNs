"""
In the main, you need to configure experiment name, decay times, decay distributions and model folder by hand.

This script generates the folder structure for the entire analyzer skripts and compute the spike rates by calling spikerate_kfold_multi().
"""
import os
from copy import deepcopy
import numpy as np
import multiprocessing
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory
from training.brightness_offset_mnist import load_mnist_with_brightness_offset_for_evaluation
from training.brightness_offset_fashionmnist import load_fashion_mnist_with_brightness_offset_for_evaluation
from training.brightness_offset_lettermnist import load_letter_mnist_with_negative_brightness_offset_for_evaluation, load_letter_mnist_with_positive_brightness_offset_for_evaluation
from training.letter_mnist import load_local_letter_mnist_for_evaluation


def extract_decay_times(dataset, decay_distributions, t_decay, x_test, dt, num_neurons, timesteps, folder_name="dt0.005", experiment="", first_timestep_to_consider=0):
    gaussian_weights = False
    use_bias = True

    if experiment.__contains__("no_bias"):
        use_bias = False

    if experiment.__contains__("gaussian_weights"):
        gaussian_weights = True


    for fold in range(0, 5):
        neuron_decay_times = np.zeros((2, num_neurons))

        for decay_distrib in decay_distributions:
            if decay_distrib != 'constant' and t_decay > 15:
                continue

            # Setup model and load weights
            # generate the model
            num_lif_units = num_neurons
            if dataset.__contains__("letter_mnist"):
                num_classes = 27
            else:
                num_classes = 10

            #simulation_time = 0.5    # duration of stimuli   [s]
            dt = 0.005               # temportal resolution  [s]
            #time_steps = int(simulation_time / dt)
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
                    decay_distribution=decay_distrib,
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
            path = f"./models/{folder_name}/{experiment}/{dataset}_time{timesteps}_dt{dt}_poisson_hidden{num_lif_units}_{decay_distrib}_decay{t_decay}ms_fold{fold}/"
            saved_model = keras.models.load_model(path, custom_objects={'use_bias': use_bias})

            model.set_weights(saved_model.get_weights())

            if experiment.__contains__("no_bias"):
                [_, w_leak] = saved_model.layers[1].get_weights()
            else:
                [_, _, w_leak] = saved_model.layers[1].get_weights()

            decay_times = 1 / w_leak # in [s]
            if decay_distrib == 'constant':
                neuron_decay_times[0] = decay_times
            elif decay_distrib == 'binned_uniform':
                neuron_decay_times[1] = decay_times
            else:
                raise ValueError(f"Invalid leak distribution in spike_rate_distribution(): {decay_distrib}")


        np.savetxt(f"./results/{dataset}/{experiment}/decay_times/{folder_name}/neuron_decay_times_decay{t_decay}ms_firststep={first_timestep_to_consider}_fold{fold}.txt", neuron_decay_times)


def multithread_wrapper(dataset, decay_distributions, decays, x_test, dt, num_neurons, timesteps, folder_name="dt0.005", experiment="", first_timestep_to_consider=0):
    # create processes
    processes = [multiprocessing.Process(target=extract_decay_times, args=[dataset, decay_distributions, t_decay, x_test, dt, num_neurons, timesteps, folder_name, experiment, first_timestep_to_consider]) 
                for t_decay in decays]

    # start the processes
    for process in processes:
        process.start()

    # wait for completion
    for process in processes:
        process.join()


if __name__ == "__main__":

    dataset = "mnist" # options: mnist, letter_mnist, fashion_mnist, cifar10 -- mnist_with_brightness_offset, letter_mnist_with_brightness_offset, fashion_mnist_with_brightness_offset
    experiment = "no_bias_gaussian_weights_128" # options: no_bias_gaussian_weights_128, no_bias_gaussian_weights_128_adj

    
    initializer_distributions = ['constant', 'binned_uniform']
    decays = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]


    # Default configuration - will be automatically reconfigured according to the name of the experiment
    num_folds = 5
    simulation_time = 0.5
    dt = 0.005
    folder_name = f"dt{dt}"

    if experiment.__contains__("512"):
        num_neurons = 512
    else:
        num_neurons = 128

    if experiment.__contains__("simulationtime"):
        simulation_time = float(experiment.split("=")[1])

    timesteps = int(simulation_time / dt)

    # For computing the spike rate we consider only the second half of the simulation
    first_timestep_to_consider = int((simulation_time * 0.5) / dt)

    # load the dataset
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    elif dataset == "mnist_with_brightness_offset":
        (x_train, y_train), (x_test, y_test) = load_mnist_with_brightness_offset_for_evaluation()
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    elif dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    elif dataset == "fashion_mnist_with_brightness_offset":
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist_with_brightness_offset_for_evaluation()
    elif dataset == "letter_mnist":
        (x_train, y_train), (x_test, y_test) = load_local_letter_mnist_for_evaluation()
    elif dataset == "letter_mnist_with_brightness_offset":
        if experiment.__contains__("neg_offset"):
            (x_train, y_train), (x_test, y_test) = load_letter_mnist_with_negative_brightness_offset_for_evaluation()
        else:
            (x_train, y_train), (x_test, y_test) = load_letter_mnist_with_positive_brightness_offset_for_evaluation()
    else:
        raise NotImplementedError(f"{dataset} needs to be loaded...")

    # Create the folder structure for saving results
    if not os.path.exists(f"./results/{dataset}"):
        os.makedirs(f"./results/{dataset}")

    if not os.path.exists(f"./results/{dataset}/{experiment}"):
        os.makedirs(f"./results/{dataset}/{experiment}")
        os.makedirs(f"./results/{dataset}/{experiment}/decay_times")
        os.makedirs(f"./results/{dataset}/{experiment}/input_output_spikes")

        os.makedirs(f"./results/{dataset}/{experiment}/decay_times/dt0.005")
        os.makedirs(f"./results/{dataset}/{experiment}/input_output_spikes/dt0.005")
        os.makedirs(f"./results/{dataset}/{experiment}/decay_times/dt0.005_untrained")
        os.makedirs(f"./results/{dataset}/{experiment}/input_output_spikes/dt0.005_untrained")



    # spike rate per neuron
    multithread_wrapper(dataset, initializer_distributions, decays, x_test, dt, num_neurons=num_neurons, timesteps=timesteps, folder_name=folder_name, experiment=experiment, first_timestep_to_consider=first_timestep_to_consider)
