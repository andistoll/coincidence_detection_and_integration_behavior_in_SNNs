import pylustrator
pylustrator.start()

import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory


def input_stimuli_vs_stimulus_window():
    decay_distrib = "constant"

    y = {}
    y_dur = {}
    for t_decay in decays:
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


    ratio_histogram_range = [0, 35]
    duration_histogram_range = [0, 175]

    plt_idx = 1
    scatter_colors = ['#ff7f0e', '#8c564b', '#7f7f7f', '#17becf']
    for key in y.keys():
        plt.subplot(2, 4, plt_idx)

        x = np.arange(duration_histogram_range[1])
        XX = np.vstack((y_dur[key], np.ones_like(y_dur[key]))).T
        p_no_offset = np.linalg.lstsq(XX[:, :-1], y[key])[0]
        plt.plot(x, p_no_offset*x, c=color, alpha=0.8, zorder=1)
        plt.scatter(y_dur[key], y[key], c=scatter_colors[plt_idx - 1], s=0.6, alpha=0.6, zorder=2)
        plt_idx += 1


    decay_distrib = "binned_uniform"

    y = {}
    y_dur = {}

    bins = []


    ratios = []
    durations = []

    for fold in range(0, num_folds):
        # Load model
        path = f"./models/{folder}/{experiment}/{dataset}_time{timesteps}_dt0.005_poisson_hidden{num_neurons}_{decay_distrib}_decay15ms_fold{fold}/"
        trained_model = keras.models.load_model(path)

        # read out decay times
        wleak = trained_model.layers[1].get_weights()[1]
        leaks = np.unique(wleak)
        decay_times = np.round(1/leaks, 3) * 1000

        i = 0
        bin_ = np.zeros_like(wleak)
        for l in sorted(leaks, reverse=True):
            bin_[np.where(wleak == l)] = i
            i += 1
        for b in bin_:
            bins.append(b)


        input_output_spikes = np.load(f"./results/{dataset}/{experiment}/input_output_spikes/{folder}/input_output_spikes_{decay_distrib}_decay15ms_{fold}.npy")

        # This ratio tells us how many input spikes this neuron needs on average in order to release an output spike
        input_output_ratio = input_output_spikes[:, 0] / input_output_spikes[:, 1]
        input_output_ratio[np.where(np.isnan(input_output_ratio))] = 0

        avg_time_till_spike = input_output_spikes[:, 2] / input_output_spikes[:, 1] * dt * 1000  # in [ms]
        for t in avg_time_till_spike:
            durations.append(t)
        for r in input_output_ratio:
            ratios.append(r)



    y[f'{decay_distrib}_15ms'] = deepcopy(ratios)
    y_dur[f'{decay_distrib}_15ms'] = deepcopy(durations)

    ratio_histogram_range = [0, 35]
    duration_histogram_range = [0, 175]


    #decays_to_plot = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]
    #bin_indices_to_plot = [0, 1, 2, 3, 4, 5, 11, 17, 25, 31]
    #decays_to_plot = [30, 90, 285, 480]
    bin_indices_to_plot = [1, 5, 17, 31]


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

            if bin_indices_to_plot.__contains__(b_idx):
                plt.subplot(2, 4, plt_idx)
                plt_idx += 1
                
                XX = np.vstack((d, np.ones_like(d))).T
                p_no_offset = np.linalg.lstsq(XX[:, :-1], r)[0]
                plt.plot(x, p_no_offset*x, c=color, alpha=0.8, zorder=1)
                plt.scatter(d, r, c='black', s=0.6, alpha=0.6, zorder=2)

            
            plt.ylim(ratio_histogram_range[0], ratio_histogram_range[1])
            plt.xlim(duration_histogram_range[0], duration_histogram_range[1])

        plt_idx += 1

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).set_size_inches(18.210000/2.54, 13.900000/2.54, forward=True)
    plt.figure(1).axes[0].set_xlim(0.0, 175)
    plt.figure(1).axes[0].set_ylim(0.0, ylim)
    plt.figure(1).axes[0].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[0].set_position([0.082702, 0.576836, 0.164909, 0.322706])
    plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
    plt.figure(1).axes[0].get_yaxis().get_label().set_text("# contrib. input spikes")
    plt.figure(1).axes[1].set_xlim(0.0, 175)
    plt.figure(1).axes[1].set_ylim(0.0, ylim)
    plt.figure(1).axes[1].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[1].set_position([0.321669, 0.576516, 0.164909, 0.322706])
    plt.figure(1).axes[2].set_xlim(0.0, 175)
    plt.figure(1).axes[2].set_ylim(0.0, ylim)
    plt.figure(1).axes[2].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[2].set_position([0.560859, 0.576516, 0.164909, 0.322706])
    plt.figure(1).axes[3].set_xlim(0.0, 175)
    plt.figure(1).axes[3].set_ylim(0.0, ylim)
    plt.figure(1).axes[3].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[3].set_position([0.800048, 0.576516, 0.164909, 0.322706])
    plt.figure(1).axes[4].set_xlim(0.0, 175)
    plt.figure(1).axes[4].set_ylim(0.0, ylim)
    plt.figure(1).axes[4].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[4].set_position([0.082479, 0.113720, 0.164909, 0.322706])
    plt.figure(1).axes[4].get_xaxis().get_label().set_text("eff. integration\n interval (ms)")
    plt.figure(1).axes[4].get_yaxis().get_label().set_text("# contrib. input spikes")
    plt.figure(1).axes[5].set_xlim(0.0, 175)
    plt.figure(1).axes[5].set_ylim(0.0, ylim)
    plt.figure(1).axes[5].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[5].set_position([0.321669, 0.113720, 0.164909, 0.322706])
    plt.figure(1).axes[5].get_xaxis().get_label().set_text("eff. integration\n interval (ms)")
    plt.figure(1).axes[6].set_xlim(0.0, 175)
    plt.figure(1).axes[6].set_ylim(0.0, ylim)
    plt.figure(1).axes[6].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[6].set_position([0.560859, 0.113720, 0.164909, 0.322706])
    plt.figure(1).axes[6].get_xaxis().get_label().set_text("eff. integration\n interval (ms)")
    plt.figure(1).axes[7].set_xlim(0.0, 175)
    plt.figure(1).axes[7].set_ylim(0.0, ylim)
    plt.figure(1).axes[7].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[7].set_position([0.800048, 0.113720, 0.164909, 0.322706])
    plt.figure(1).axes[7].get_xaxis().get_label().set_text("eff. integration\n interval (ms)")
    plt.figure(1).text(0.5, 0.5, f'{dataset_name}', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
    plt.figure(1).texts[0].set_position([0.434391, 0.976695])
    plt.figure(1).text(0.5, 0.5, 'constant', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[1].new
    plt.figure(1).texts[1].set_position([0.472881, 0.936389])
    plt.figure(1).text(0.5, 0.5, 'binned uniform', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[2].new
    plt.figure(1).texts[2].set_position([0.439889, 0.485828])
    plt.figure(1).text(0.5, 0.5, 'decay 30ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
    plt.figure(1).texts[3].set_position([0.109972, 0.908319])
    plt.figure(1).text(0.5, 0.5, 'decay 90ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
    plt.figure(1).texts[4].set_position([0.351911, 0.908319])
    plt.figure(1).text(0.5, 0.5, 'decay 285ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
    plt.figure(1).texts[5].set_position([0.588352, 0.908319])
    plt.figure(1).text(0.5, 0.5, 'decay 480ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
    plt.figure(1).texts[6].set_position([0.824792, 0.908319])
    plt.figure(1).text(0.5, 0.5, 'decay 30ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[7].new
    plt.figure(1).texts[7].set_position([0.109972, 0.448401])
    plt.figure(1).text(0.5, 0.5, 'decay 90ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[8].new
    plt.figure(1).texts[8].set_position([0.351911, 0.448401])
    plt.figure(1).text(0.5, 0.5, 'decay 285ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[9].new
    plt.figure(1).texts[9].set_position([0.588352, 0.448401])
    plt.figure(1).text(0.5, 0.5, 'decay 480ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[10].new
    plt.figure(1).texts[10].set_position([0.824792, 0.448401])
    plt.figure(1).text(0.5, 0.5, 'a', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[11].new
    plt.figure(1).texts[11].set_position([0.054986, 0.936389])
    plt.figure(1).text(0.5, 0.5, 'b', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[12].new
    plt.figure(1).texts[12].set_position([0.054986, 0.485828])
    #% end: automatic generated code from pylustrator
    #plt.show()

    if not os.path.exists("./results/figures"):
        os.makedirs("./results/figures")

    plt.savefig(f"./results/figures/{dataset}_input_vs_window_scatter.pdf")
    plt.close("all")



def input_stimuli_vs_stimulus_window_no_header():
    decay_distrib = "constant"

    y = {}
    y_dur = {}
    for t_decay in decays:
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


    ratio_histogram_range = [0, 35]
    duration_histogram_range = [0, 175]

    plt_idx = 1
    scatter_colors = ['#ff7f0e', '#8c564b', '#7f7f7f', '#17becf']
    for key in y.keys():
        plt.subplot(2, 4, plt_idx)

        x = np.arange(duration_histogram_range[1])
        XX = np.vstack((y_dur[key], np.ones_like(y_dur[key]))).T
        p_no_offset = np.linalg.lstsq(XX[:, :-1], y[key])[0]
        plt.plot(x, p_no_offset*x, c=color, alpha=0.8, zorder=1)
        plt.scatter(y_dur[key], y[key], c=scatter_colors[plt_idx - 1], s=0.6, alpha=0.6, zorder=2)
        plt_idx += 1


    decay_distrib = "binned_uniform"

    y = {}
    y_dur = {}

    bins = []


    ratios = []
    durations = []

    for fold in range(0, num_folds):
        # Load model
        path = f"./models/{folder}/{experiment}/{dataset}_time{timesteps}_dt0.005_poisson_hidden{num_neurons}_{decay_distrib}_decay15ms_fold{fold}/"
        trained_model = keras.models.load_model(path)

        # read out decay times
        wleak = trained_model.layers[1].get_weights()[1]
        leaks = np.unique(wleak)
        decay_times = np.round(1/leaks, 3) * 1000

        i = 0
        bin_ = np.zeros_like(wleak)
        for l in sorted(leaks, reverse=True):
            bin_[np.where(wleak == l)] = i
            i += 1
        for b in bin_:
            bins.append(b)


        input_output_spikes = np.load(f"./results/{dataset}/{experiment}/input_output_spikes/{folder}/input_output_spikes_{decay_distrib}_decay15ms_{fold}.npy")

        # This ratio tells us how many input spikes this neuron needs on average in order to release an output spike
        input_output_ratio = input_output_spikes[:, 0] / input_output_spikes[:, 1]
        input_output_ratio[np.where(np.isnan(input_output_ratio))] = 0

        avg_time_till_spike = input_output_spikes[:, 2] / input_output_spikes[:, 1] * dt * 1000  # in [ms]
        for t in avg_time_till_spike:
            durations.append(t)
        for r in input_output_ratio:
            ratios.append(r)



    y[f'{decay_distrib}_15ms'] = deepcopy(ratios)
    y_dur[f'{decay_distrib}_15ms'] = deepcopy(durations)

    ratio_histogram_range = [0, 35]
    duration_histogram_range = [0, 175]


    #decays_to_plot = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]
    #bin_indices_to_plot = [0, 1, 2, 3, 4, 5, 11, 17, 25, 31]
    #decays_to_plot = [30, 90, 285, 480]
    bin_indices_to_plot = [1, 5, 17, 31]


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

            if bin_indices_to_plot.__contains__(b_idx):
                plt.subplot(2, 4, plt_idx)
                plt_idx += 1
                
                XX = np.vstack((d, np.ones_like(d))).T
                p_no_offset = np.linalg.lstsq(XX[:, :-1], r)[0]
                plt.plot(x, p_no_offset*x, c=color, alpha=0.8, zorder=1)
                plt.scatter(d, r, c='black', s=0.6, alpha=0.6, zorder=2)

            
            plt.ylim(ratio_histogram_range[0], ratio_histogram_range[1])
            plt.xlim(duration_histogram_range[0], duration_histogram_range[1])

        plt_idx += 1

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).set_size_inches(18.210000/2.54, 13.900000/2.54, forward=True)
    plt.figure(1).axes[0].set_xlim(0.0, 175)
    plt.figure(1).axes[0].set_ylim(0.0, ylim)
    plt.figure(1).axes[0].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[0].set_position([0.082702, 0.576836, 0.164909, 0.322706])
    plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
    plt.figure(1).axes[0].get_yaxis().get_label().set_text("# contrib. input spikes")
    plt.figure(1).axes[1].set_xlim(0.0, 175)
    plt.figure(1).axes[1].set_ylim(0.0, ylim)
    plt.figure(1).axes[1].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[1].set_position([0.321669, 0.576516, 0.164909, 0.322706])
    plt.figure(1).axes[2].set_xlim(0.0, 175)
    plt.figure(1).axes[2].set_ylim(0.0, ylim)
    plt.figure(1).axes[2].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[2].set_position([0.560859, 0.576516, 0.164909, 0.322706])
    plt.figure(1).axes[3].set_xlim(0.0, 175)
    plt.figure(1).axes[3].set_ylim(0.0, ylim)
    plt.figure(1).axes[3].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[3].set_position([0.800048, 0.576516, 0.164909, 0.322706])
    plt.figure(1).axes[4].set_xlim(0.0, 175)
    plt.figure(1).axes[4].set_ylim(0.0, ylim)
    plt.figure(1).axes[4].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[4].set_position([0.082479, 0.113720, 0.164909, 0.322706])
    plt.figure(1).axes[4].get_xaxis().get_label().set_text("eff. integration\n interval (ms)")
    plt.figure(1).axes[4].get_yaxis().get_label().set_text("# contrib. input spikes")
    plt.figure(1).axes[5].set_xlim(0.0, 175)
    plt.figure(1).axes[5].set_ylim(0.0, ylim)
    plt.figure(1).axes[5].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[5].set_position([0.321669, 0.113720, 0.164909, 0.322706])
    plt.figure(1).axes[5].get_xaxis().get_label().set_text("eff. integration\n interval (ms)")
    plt.figure(1).axes[6].set_xlim(0.0, 175)
    plt.figure(1).axes[6].set_ylim(0.0, ylim)
    plt.figure(1).axes[6].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[6].set_position([0.560859, 0.113720, 0.164909, 0.322706])
    plt.figure(1).axes[6].get_xaxis().get_label().set_text("eff. integration\n interval (ms)")
    plt.figure(1).axes[7].set_xlim(0.0, 175)
    plt.figure(1).axes[7].set_ylim(0.0, ylim)
    plt.figure(1).axes[7].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
    plt.figure(1).axes[7].set_position([0.800048, 0.113720, 0.164909, 0.322706])
    plt.figure(1).axes[7].get_xaxis().get_label().set_text("eff. integration\n interval (ms)")
    plt.figure(1).text(0.5, 0.5, f'{dataset_name}', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
    plt.figure(1).texts[0].set_position([0.434391, 0.976695])
    plt.figure(1).text(0.5, 0.5, 'constant', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[1].new
    plt.figure(1).texts[1].set_position([0.472881, 0.936389])
    plt.figure(1).text(0.5, 0.5, 'binned uniform', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[2].new
    plt.figure(1).texts[2].set_position([0.439889, 0.485828])
    plt.figure(1).text(0.5, 0.5, 'decay 30ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
    plt.figure(1).texts[3].set_position([0.109972, 0.908319])
    plt.figure(1).text(0.5, 0.5, 'decay 90ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
    plt.figure(1).texts[4].set_position([0.351911, 0.908319])
    plt.figure(1).text(0.5, 0.5, 'decay 285ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
    plt.figure(1).texts[5].set_position([0.588352, 0.908319])
    plt.figure(1).text(0.5, 0.5, 'decay 480ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
    plt.figure(1).texts[6].set_position([0.824792, 0.908319])
    plt.figure(1).text(0.5, 0.5, 'decay 30ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[7].new
    plt.figure(1).texts[7].set_position([0.109972, 0.448401])
    plt.figure(1).text(0.5, 0.5, 'decay 90ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[8].new
    plt.figure(1).texts[8].set_position([0.351911, 0.448401])
    plt.figure(1).text(0.5, 0.5, 'decay 285ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[9].new
    plt.figure(1).texts[9].set_position([0.588352, 0.448401])
    plt.figure(1).text(0.5, 0.5, 'decay 480ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[10].new
    plt.figure(1).texts[10].set_position([0.824792, 0.448401])
    plt.figure(1).text(0.5, 0.5, 'a', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[11].new
    plt.figure(1).texts[11].set_position([0.054986, 0.936389])
    plt.figure(1).text(0.5, 0.5, 'b', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[12].new
    plt.figure(1).texts[12].set_position([0.054986, 0.485828])
    #% end: automatic generated code from pylustrator
    #plt.show()

    if not os.path.exists("./results/figures"):
        os.makedirs("./results/figures")

    plt.savefig(f"./results/figures/{dataset}_input_vs_window_scatter_no_header.pdf")
    plt.close("all")


if __name__ == "__main__":
    num_folds = 5
    simulation_time = 0.5
    dt = 0.005
    timesteps = int(simulation_time / dt)
    first_timestep_to_consider = int((simulation_time * 0.5) / dt)
    folder = f"dt{dt}"
    

    bin_no = 75


    experiment = "no_bias_gaussian_weights_128"
    decays = [30, 90, 285, 480] # only plot w.r.t. these decay times

    if not os.path.exists("./results/figures"):
        os.makedirs("./results/figures")

    num_neurons = 128
    distribution_range = f" [15, {decays[-1]}]ms"

    for dataset in  ["mnist", "fashion_mnist", "cifar10", "letter_mnist"]:

        if dataset == "mnist":
            dataset_name = "        MNIST"
            ylim = 20.0
            color = '#1f77b4'
            #input_stimuli_vs_stimulus_window_no_header()
        if dataset == "mnist_with_brightness_offset":
            dataset_name = "MNIST adj."
        if dataset == "fashion_mnist":
            dataset_name = "Fashion-MNIST"
            ylim = 25.0
            color = '#9467bd'
        if dataset == "cifar10":
            dataset_name = "      CIFAR-10"
            ylim = 35.0
            color = '#2ca02c'
        if dataset == "letter_mnist":
            dataset_name = "EMNIST/Letters"
            ylim = 20.0
            color = 'darkgoldenrod'

        input_stimuli_vs_stimulus_window()
