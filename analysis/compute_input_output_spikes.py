from copy import deepcopy
import os
import numpy as np
import multiprocessing
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import gc
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory
from training.brightness_offset_mnist import load_mnist_with_brightness_offset_for_evaluation
from training.brightness_offset_fashionmnist import load_fashion_mnist_with_brightness_offset_for_evaluation
from training.brightness_offset_lettermnist import load_letter_mnist_with_negative_brightness_offset_for_evaluation, load_letter_mnist_with_positive_brightness_offset_for_evaluation
from training.letter_mnist import load_local_letter_mnist_for_evaluation


def compute_input_output_spike_ratio(dataset, folder, experiment, decay_distrib, t_decay, num_neurons, timesteps, num_folds=5):
    # load the dataset
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    elif dataset == "mnist_with_brightness_offset":
        (x_train, y_train), (x_test, y_test) = load_mnist_with_brightness_offset_for_evaluation()
    elif dataset == "letter_mnist":
        (x_train, y_train), (x_test, y_test) = load_local_letter_mnist_for_evaluation()
    elif dataset == "letter_mnist_with_brightness_offset":
        if experiment.__contains__("neg_offset"):
            (x_train, y_train), (x_test, y_test) = load_letter_mnist_with_negative_brightness_offset_for_evaluation()
        else:
            (x_train, y_train), (x_test, y_test) = load_letter_mnist_with_positive_brightness_offset_for_evaluation()
    elif dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    elif dataset == "fashion_mnist_with_brightness_offset":
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist_with_brightness_offset_for_evaluation()
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError(f"{dataset} needs to be loaded...")

    use_bias = True
    gaussian_weights = False

    if experiment.__contains__("no_bias"):
        use_bias = False

    if experiment.__contains__("gaussian_weights"):
        gaussian_weights = True


    if decay_distrib != 'constant' and t_decay > 15:
        return

    for fold in range(num_folds):
        print(f"fold {fold}")

        # Ratio of input spikes to output spikes
        input_output_spikes = np.zeros((num_neurons, 3))
        spike_threshold = 0.9999  # a bit smaller than 1 because of potential inaccuracies

        # Load weights and set up model outputting Vm
        num_lif_units = num_neurons
        path = f"./models/{folder}/{experiment}/{dataset}_time{timesteps}_dt0.005_poisson_hidden{num_lif_units}_{decay_distrib}_decay{t_decay}ms_fold{fold}/"
        trained_model = keras.models.load_model(path)
        
        if dataset.__contains__("letter_mnist"):
                num_classes = 27
        else:
            num_classes = 10
        
        simulation_time = 0.5
        dt = 0.005
        time_steps = int(simulation_time / dt)
        model = keras.models.Sequential([
            keras.Input(x_test.shape[1:]),
            IntensityToPoissonSpiking(time_steps, 1/255, dt=dt),
            DenseLIF(num_lif_units,
                gaussian_weights=gaussian_weights,
                use_bias=use_bias, 
                surrogate="flat", 
                return_potential=False,
                dt=dt,
                decay_distribution=decay_distrib,
                trainable_leak=False, 
                t_decay=t_decay/1000),
            DenseLIFNoSpike(num_classes),
        ])

        for i in range(len(model.layers)):
            model.layers[i].set_weights(trained_model.layers[i].get_weights())

            if experiment.__contains__("no_bias"):
                [w_input, w_leak] = model.layers[1].get_weights()
            else:
                [w_input, bias, w_leak] = model.layers[1].get_weights()


        for img in range(1000, 2000): # we only consider 1000 test images

            # 1) load test image
            test_image = np.expand_dims(x_test[img] , axis=0)

            # Functions for looking into layers
            get_poisson_layer_output = keras.backend.function([model.layers[0].input], [model.layers[0].output])
            get_lif_layer_output = keras.backend.function([model.layers[1].input], [model.layers[1].output])

            # 2) infere image: extract Poisson spikes and LIF spikes
            poisson = get_poisson_layer_output([test_image])[0]
            lif = get_lif_layer_output([poisson])[0]
            lif = lif[0]
            poisson = poisson[0]

            for neuron in range(lif.shape[1]):     # shape: 128

                prev_spike = 0

                for step in range(lif.shape[0]):   # shape: 100

                    activity = lif[step, neuron]

                    if activity == 0:   # no spike
                        continue
                    elif activity == 1: # spike
                        # We found a spike and now we have to backtrack the inputs coming in at timestep t_lif_spike down to 0
                    
                        input_output_spikes[neuron][1] += 1  # increase the count of output spikes

                        # 1) extract timestep of this spike
                        t_lif_spike = step

                        # read out the input stimulus at the time of the spike
                        input_stimulus = 0 #poisson[t_lif_spike, :] * w_input[:, neuron] # shape of weights: (784, 128)

                        #print(t_lif_spike)
                        # 2) extract the input weights this neuron is sensitive to
                        leak_power = 0
                        for t in reversed(range(prev_spike, t_lif_spike + 1)):
                            # iterate from time of spike - 1 down to the previous spike
                            input_stimulus += poisson[t, :] * w_input[:, neuron] * ((1 - w_leak[neuron] * dt)**leak_power)
                            leak_power +=1

                            if sum(input_stimulus) >= spike_threshold:
                                # compute the amount of input spikes
                                input_output_spikes[neuron][0] += np.where(poisson[t:t_lif_spike+1, :] == 1)[0].size

                                # store the amount of timesteps it took to stimulate this spike
                                input_output_spikes[neuron][2] += t_lif_spike - t + 1 # +1 because t_lif_spike is also included
                                break
                            else:
                                if t == 0:
                                    print(f"t=0, prev_spike={prev_spike}")
                                    print("stimulus =", sum(input_stimulus))
                                    raise ValueError("the input simuli have not been high enough to release a spike, but a spike was detected...")

                                # go one step further back in time
                                pass

                    # update the timestamp of the previous spike, so that we have a limit for our backtracking
                    prev_spike = t_lif_spike


        np.save(f"./results/{dataset}/{experiment}/input_output_spikes/{folder}/input_output_spikes_{decay_distrib}_decay{t_decay}ms_{fold}.npy", input_output_spikes)
        keras.backend.clear_session()

    


if __name__ == "__main__":

    dataset = "mnist" # options: mnist, letter_mnist, fashion_mnist, cifar10 -- mnist_with_brightness_offset, letter_mnist_with_brightness_offset, fashion_mnist_with_brightness_offset
    experiment = "no_bias_gaussian_weights_128" # options: no_bias_gaussian_weights_128, no_bias_gaussian_weights_128_adj
    
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


    initializer_distributions = ['binned_uniform', 'constant']
    decays = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]


    # Default configuration - will be automatically reconfigured according to the name of the experiment
    num_folds = 5
    num_neurons = 128
    simulation_time = 0.5
    dt = 0.005
    timesteps = int(simulation_time / dt)
    folder = f"dt{dt}"

    try:
        process_list = []

	# process every decay time in a new thread
        for distrib in initializer_distributions:
            processes = [multiprocessing.Process(target=compute_input_output_spike_ratio, args=[dataset, folder, experiment, distrib, decay, num_neurons, timesteps]) for decay in decays]
            # queue the processes
            for process in processes:
                process_list.append(process)
                #process.start()


        # start the processes
        for process in process_list:
            process.start()
        # wait for completion
        for process in process_list:
            process.join()

    except Exception as e:
        print(e)
