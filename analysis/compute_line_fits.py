from copy import deepcopy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory
import pandas as pd

        

def operation_mode_scatter_line_fits(dataset, initializer_distributions, decays, slopes_binned, decays_binned, dt, bin_no=25, folder="", experiment=""):
    num_folds=5
    y = {}
    y_dur = {}

    slopes = []
    y_intercepts = []
    slopes2 = []

    for t_decay in decays:
        for decay_distrib in initializer_distributions:
            if decay_distrib != 'constant' and t_decay > 15:
                continue

            #if decay_distrib == 'constant' and t_decay == 15:
            #    continue

            ratios = []
            durations = []

            for fold in range(0, num_folds):
                input_output_spikes = np.load(f"./results/{dataset}/{experiment}/input_output_spikes/{folder}/input_output_spikes_{decay_distrib}_decay{t_decay}ms_{fold}.npy")

                # This ratio tells us how many input spikes this neuron needs on average in order to release an output spike
                input_output_ratio = input_output_spikes[:, 0] / input_output_spikes[:, 1]
                input_output_ratio[np.where(np.isnan(input_output_ratio))] = 0

                avg_time_till_spike = input_output_spikes[:, 2] / input_output_spikes[:, 1] * dt * 1000  # in [ms]
                for t in avg_time_till_spike:
                    durations.append(t)
                for r in input_output_ratio:
                    ratios.append(r)


            y[f'{decay_distrib}_{t_decay}ms'] = deepcopy(ratios)
            y_dur[f'{decay_distrib}_{t_decay}ms'] = deepcopy(durations)


    for key in y.keys():

	    # fit line through data
        m, b = np.polyfit(y_dur[key], y[key], 1)
        slopes.append(m)
        y_intercepts.append(b)

	    # line fit through data and coordinate origin
        XX = np.vstack((y_dur[key], np.ones_like(y_dur[key]))).T
        p_no_offset = np.linalg.lstsq(XX[:, :-1], y[key])[0]
        slopes2.append(p_no_offset[0])
        y_fit = np.dot(p_no_offset, XX[:, :-1].T)

    return slopes2, decays


def operation_mode_scatter_line_fits_binned(dataset, initializer_distributions, decays, dt, bin_no=25, folder="", experiment=""):
    num_folds=5
    y = {}
    y_dur = {}

    slopes = []
    y_intercepts = []
    slopes2 = []

    bins = []
    decay_times = []

    for t_decay in decays:
        for decay_distrib in initializer_distributions:
            if decay_distrib != 'constant' and t_decay > 15:
                continue
            if decay_distrib == 'constant':
                continue

            ratios = []
            durations = []

            for fold in range(0, num_folds):
                # Load model
                path = f"./models/{folder}/{experiment}/{dataset}_time{timesteps}_dt0.005_poisson_hidden{num_neurons}_{decay_distrib}_decay{t_decay}ms_fold{fold}/"
                trained_model = keras.models.load_model(path)

                # read out decay times
                wleak = trained_model.layers[1].get_weights()[1]
                leaks = np.unique(wleak)
                decay_times = sorted(np.round(1/leaks, 3) * 1000)

                i = 0
                bin_ = np.zeros_like(wleak)
                for l in sorted(leaks, reverse=True):
                    bin_[np.where(wleak == l)] = i
                    i += 1
                for b in bin_:
                    bins.append(b)


                input_output_spikes = np.load(f"./results/{dataset}/{experiment}/input_output_spikes/{folder}/input_output_spikes_{decay_distrib}_decay{t_decay}ms_{fold}.npy")

                # This ratio tells us how many input spikes this neuron needs on average in order to release an output spike
                input_output_ratio = input_output_spikes[:, 0] / input_output_spikes[:, 1]
                input_output_ratio[np.where(np.isnan(input_output_ratio))] = 0

                avg_time_till_spike = input_output_spikes[:, 2] / input_output_spikes[:, 1] * dt * 1000  # in [ms]
                for t in avg_time_till_spike:
                    durations.append(t)
                for r in input_output_ratio:
                    ratios.append(r)



            y[f'{decay_distrib}_{t_decay}ms'] = deepcopy(ratios)
            y_dur[f'{decay_distrib}_{t_decay}ms'] = deepcopy(durations)


    decays_to_plot = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]
    bin_indices_to_plot = [0, 1, 2, 3, 4, 5, 11, 17, 25, 31]

    for key in y.keys():
        avg_over_bins = 1  # TODO: With this parameter, line fitting in more than 1 bin is possible
        unique_bins = sorted(np.unique(bins))

        for b_idx in range(0, len(unique_bins), avg_over_bins):
            indices = np.where(bins == unique_bins[b_idx])[0]
            for i in range(1, avg_over_bins):
                try:
                    indices = np.hstack((indices, np.where(bins == unique_bins[b_idx + i])[0]))
                except IndexError:
                    pass

            d = np.asarray(y_dur[key])[indices]
            r = np.asarray(y[key])[indices]

            # line fit through data
            m, t = np.polyfit(d, r, 1)
            slopes.append(m)
            y_intercepts.append(t)

            # line fit through data and coordinate origin
            XX = np.vstack((d, np.ones_like(d))).T
            p_no_offset = np.linalg.lstsq(XX[:, :-1], r)[0]
            slopes2.append(p_no_offset[0])
            y_fit = np.dot(p_no_offset, XX[:, :-1].T)

    return slopes2, decay_times



if __name__ == "__main__":

    dataset = "mnist" # options: mnist, letter_mnist, fashion_mnist, cifar10 -- mnist_with_brightness_offset, letter_mnist_with_brightness_offset, fashion_mnist_with_brightness_offset
    experiment = "no_bias_gaussian_weights_128" # options: no_bias_gaussian_weights_128, no_bias_gaussian_weights_128_adj


    decays = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]

    # Default configuration - will be automatically reconfigured according to the name of the experiment
    num_folds = 5
    num_neurons = 128
    simulation_time = 0.5
    dt = 0.005
    timesteps = int(simulation_time / dt)
    folder = f"dt{dt}"


    # Compute line fits in operation mode scatter plots
    slopes_binned, decays_binned = operation_mode_scatter_line_fits_binned(dataset, ['binned_uniform'], decays, dt=dt, bin_no=128, folder=folder, experiment=experiment)
    slopes_const, decays_const = operation_mode_scatter_line_fits(dataset, ['constant'], decays, slopes_binned, decays_binned, dt=dt, bin_no=128, folder=folder, experiment=experiment)

    operation_mode_scatter_line_fits(dataset, ['binned_uniform'], decays, slopes_binned, decays_binned, dt=dt,bin_no=128, folder=folder, experiment=experiment)


    # Store the computed line fits
    line_fits_binned = pd.DataFrame(data={"decays": decays_binned, "slopes": slopes_binned}, columns=["decays", "slopes"])
    line_fits_binned.to_csv(f"./results/{dataset}/{experiment}/line_fits_binned.csv")
    line_fits_const = pd.DataFrame(data={"decays": decays_const, "slopes": slopes_const}, columns=["decays", "slopes"])
    line_fits_const.to_csv(f"./results/{dataset}/{experiment}/line_fits_const.csv")
