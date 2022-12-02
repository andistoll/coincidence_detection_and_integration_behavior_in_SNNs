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


def io_vs_tdecay():
    decay_distrib = "constant"

    y = {}
    y_dec = {}

    plt_idx = 1
    for io_or_tts in ["io", "tts"]:
        for t_decay in decays:
            

            ratios, durations = [], []
            decs = []

            for fold in range(0, num_folds):
                neuron_decay_times = np.loadtxt(f"./results/{dataset}/{experiment}/decay_times/{folder}/neuron_decay_times_decay{t_decay}ms_firststep={first_timestep_to_consider}_fold{fold}.txt")
                idx = 0 # constant

                for dec in neuron_decay_times[idx]:
                    decs.append(dec * 1000)
                
                input_output_spikes = np.load(f"./results/{dataset}/{experiment}/input_output_spikes/{folder}/input_output_spikes_{decay_distrib}_decay{t_decay}ms_{fold}.npy")


                # This ratio tells us how many input spikes this neuron needs on average in order to release an output spike
                input_output_ratio = input_output_spikes[:, 0] / input_output_spikes[:, 1]
                input_output_ratio[np.where(np.isnan(input_output_ratio))] = 0

                avg_time_till_spike = input_output_spikes[:, 2] / input_output_spikes[:, 1] * dt * 1000  # in [ms]
                for t in avg_time_till_spike:
                    durations.append(t)
                for r in input_output_ratio:
                    ratios.append(r)

            if io_or_tts == "io":
                y[f'{decay_distrib}_{t_decay}ms'] = deepcopy(ratios)
            elif io_or_tts == "tts":
                y[f'{decay_distrib}_{t_decay}ms'] = deepcopy(durations)
            else:
                raise ValueError("io_or_tts")

            y_dec[f'{decay_distrib}_{t_decay}ms'] = deepcopy(decs)

        if io_or_tts == "io":
            ratio_histogram_range = [0, 35]
        else:
            ratio_histogram_range = [0, 125]

        duration_histogram_range = [0, 500]


        means_const, stds_const = {}, {}
        plt.subplot(1, 8, plt_idx)
        plt_idx += 1

        decs, ratios = [], []
        for key in y.keys():
            if not as_histo:
                means_const[key] = np.mean(y[key])
                stds_const[key] = np.std(y[key])
                continue

            for i in y_dec[key]:
                decs.append(i)
            for i in y[key]:
                ratios.append(i)

        if as_histo:    
            plt.hist2d(np.asarray(decs), np.asarray(ratios), bins=bin_no, range=(duration_histogram_range, ratio_histogram_range), cmap='gray_r')
            if io_or_tts == "io":
                plt.ylabel(f"# contrib. input spikes")
            else:
                plt.ylabel("eff. integ. interval")
            plt.xlabel(f"decay time [ms]")


        if means_const != {}:
            counter = 0
            decs, means, stds = [], [], []
            for key in means_const.keys():
                decs.append(decays[counter])
                means.append(means_const[key])
                stds.append(stds_const[key])

                plt.errorbar(decays[counter], means_const[key], yerr=stds_const[key], fmt="o", ecolor='black', markersize=4, elinewidth=1, label=f'{decays[counter]} ms')
                counter += 1


            if io_or_tts == "io":
                plt.ylabel(f"# contrib. input spikes")
            else:
                plt.ylabel("eff. integ. interval")

            plt.xlabel(f"decay time [ms]")
            plt.xlim(duration_histogram_range[0], duration_histogram_range[1])
            if io_or_tts == "io":
                plt.ylim(ratio_histogram_range[0], io_lim)
                if dataset == "mnist":
                    plt.yticks([0.0, 2.5, 5.0, 7.5, 10.0, 12.5], ["0", "", "5", "", "10", ""])
                if dataset == "letter_mnist":
                    plt.yticks([0.0, 2.5, 5.0, 7.5, 10.0], ["0", "", "5", "", "10"])
                if dataset == "cifar10":
                    plt.yticks([0.0, 5.0, 10.0, 15, 20, 25, 30], ["0", "", "10", "", "20", "", "30"])
            else:
                plt.ylim(ratio_histogram_range[0], tts_lim)

            
            if plt_idx != 3:
                plt.legend(loc='lower right', fontsize=5, ncol=2)


    

    decay_distrib = "binned_uniform"
    t_decay = 15

    y = {}
    y_dec = {}
    ratios, durations = [], []
    decs = []

    for io_or_tts in ["io", "tts"]:
        for fold in range(0, num_folds):
            neuron_decay_times = np.loadtxt(f"./results/{dataset}/{experiment}/decay_times/{folder}/neuron_decay_times_decay{t_decay}ms_firststep={first_timestep_to_consider}_fold{fold}.txt")

            idx = 1 # load binned_uniform data

            for dec in neuron_decay_times[idx]:
                decs.append(dec * 1000)
            
            input_output_spikes = np.load(f"./results/{dataset}/{experiment}/input_output_spikes/{folder}/input_output_spikes_{decay_distrib}_decay{t_decay}ms_{fold}.npy")

            # This ratio tells us how many input spikes this neuron needs on average in order to release an output spike
            input_output_ratio = input_output_spikes[:, 0] / input_output_spikes[:, 1]
            input_output_ratio[np.where(np.isnan(input_output_ratio))] = 0

            avg_time_till_spike = input_output_spikes[:, 2] / input_output_spikes[:, 1] * dt * 1000  # in [ms]
            for t in avg_time_till_spike:
                durations.append(t)
            for r in input_output_ratio:
                ratios.append(r)

            if io_or_tts == "io":
                y[f'{decay_distrib}_{t_decay}ms'] = deepcopy(ratios)
            elif io_or_tts == "tts":
                y[f'{decay_distrib}_{t_decay}ms'] = deepcopy(durations)
            else:
                raise ValueError("io_or_tts")

            y_dec[f'{decay_distrib}_{t_decay}ms'] = deepcopy(decs)

    
        if io_or_tts == "io":
            ratio_histogram_range = [0, 30]
        else:
            ratio_histogram_range = [0, 125]

        duration_histogram_range = [0, 500]


        for key in y.keys():

            label = str(key).split('_15')[0] + distribution_range
            plt.subplot(1, 8, plt_idx)
            plt_idx += 1

            if as_histo:
                plt.hist2d(y_dec[key], y[key], bins=bin_no, range=(duration_histogram_range, ratio_histogram_range), cmap='gray_r')
                plt.title(label)
                if io_or_tts == "io":
                    plt.ylabel(f"# contrib. input spikes")
                else:
                    plt.ylabel("eff. integ. interval")
                plt.xlabel(f"decay time [ms]")

            else: # plot mean and std per bin instead of 2D histogram
                t_dec = sorted(np.unique(y_dec[key]))

                means, stds = [], []
                for dec in t_dec:
                    indices = np.where(y_dec[key] == dec)[0]
                    ratios = []
                    for i in indices:
                        ratios.append(y[key][i])
                    means.append(np.mean(ratios))
                    stds.append(np.std(ratios))

                t_dec = [int(np.round(i, 0)) for i in t_dec]

                plt.errorbar(t_dec, means, yerr=stds, fmt="o", ecolor='black', markersize=4, elinewidth=1)
                plt.xlabel(f"decay time [ms]")

                plt.xlim(duration_histogram_range[0], duration_histogram_range[1])
                if io_or_tts == "io":
                    plt.ylim(ratio_histogram_range[0], io_lim)
                    if dataset == "mnist":
                        plt.yticks([0.0, 2.5, 5.0, 7.5, 10.0, 12.5], ["0", "", "5", "", "10", ""])
                    if dataset == "letter_mnist":
                        plt.yticks([0.0, 2.5, 5.0, 7.5, 10.0], ["0", "", "5", "", "10"])
                    if dataset == "cifar10":
                        plt.yticks([0.0, 5.0, 10.0, 15, 20, 25, 30], ["0", "", "10", "", "20", "", "30"])
                else:
                    plt.ylim(ratio_histogram_range[0], tts_lim)


                plt.scatter(np.array([-10]), np.array([-10]), marker='o', label="bins", s=12)
                if plt_idx != 5:
                    plt.legend(loc='lower right', fontsize=8)

            idx += 1

    plt.subplot(1, 8, plt_idx)
    from tf_spiking.custom_initializers import BinnedUniformDecayInitializer
    init = BinnedUniformDecayInitializer(distribution_type='binned_uniform', bin_step_size=None, longest_t_decay=0.48, t_decay=0.015, show_distribution=False)
    tensor = init.__call__((128, ))
    print(tensor)
    d = 1 / tensor

    plt.hist(d * 1000, bins=128, range=[15, 480])
    plt.xlim(0, 500)
    plt.ylim(0, 4.2)
    plt.xticks([15, 480], fontsize=6)
    plt.yticks([0, 4], fontsize=6)
    plt.tick_params(axis='both', which='both',
                        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                        bottom=True, top=False, left=True, right=False)
    plt_idx += 1

    idx = 0
    scatter_colors = ['#ff7f0e', '#e377c2', '#17becf']
    for i in [30, 180, 480]:
        plt.subplot(1, 8, plt_idx)
        plt_idx += 1
        plt.plot([i, i], [0, 128], c=scatter_colors[idx])
        idx += 1
        if i == 30:
            plt.xlim(i-10, i+30)
        elif i == 180:
            plt.xlim(i-20, i+20)
        else:
            plt.xlim(i-30, i+10)
        plt.ylim(0, 140)
        plt.xticks([i], fontsize=6)
        plt.yticks([0, 128], fontsize=6)
        if i == 30:
            plt.tick_params(axis='both', which='both',
                                labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                                bottom=True, top=False, left=True, right=False)
        else:
            plt.tick_params(axis='both', which='both',
                            labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                            bottom=True, top=False, left=False, right=False)



    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).set_size_inches(20.210000/2.54, 6.950000/2.54, forward=True)
    plt.figure(1).axes[0].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
    plt.figure(1).axes[0].set_position([0.064379, 0.165845, 0.164958, 0.576851])
    plt.figure(1).axes[0].get_xaxis().get_label().set_text("decay time (ms)")
    plt.figure(1).axes[0].get_yaxis().get_label().set_text("# contrib. input spikes")
    plt.figure(1).axes[1].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
    plt.figure(1).axes[1].set_position([0.311989, 0.165845, 0.164958, 0.576851])
    plt.figure(1).axes[1].get_xaxis().get_label().set_text("decay time (ms)")
    plt.figure(1).axes[1].get_yaxis().get_label().set_text("integ. interval (ms)")
    plt.figure(1).axes[2].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
    plt.figure(1).axes[2].set_position([0.574456, 0.165845, 0.164958, 0.576851])
    plt.figure(1).axes[2].get_xaxis().get_label().set_text("decay time (ms)")
    plt.figure(1).axes[2].get_yaxis().get_label().set_text("# contrib. input spikes")
    plt.figure(1).axes[3].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
    plt.figure(1).axes[3].set_position([0.819591, 0.166116, 0.164958, 0.576851])
    plt.figure(1).axes[3].get_xaxis().get_label().set_text("decay time (ms)")
    plt.figure(1).axes[3].get_yaxis().get_label().set_text("integ. interval (ms)")
    plt.figure(1).axes[4].set_xlim(-10.0, 510.0)
    plt.figure(1).axes[4].get_yaxis().set_label_position("right")
    plt.figure(1).axes[4].set_position([0.880958, 0.245347, 0.099044, 0.135782])
    plt.figure(1).axes[4].get_xaxis().get_label().set_text(" ")
    plt.figure(1).axes[4].get_yaxis().get_label().set_text("# LIF")
    plt.figure(1).axes[5].set_position([0.378844, 0.245162, 0.024761, 0.135560])
    plt.figure(1).axes[5].get_yaxis().get_label().set_text("")
    plt.figure(1).axes[6].set_position([0.408557, 0.245162, 0.029713, 0.135560])
    plt.figure(1).axes[7].get_yaxis().set_label_position("right")
    plt.figure(1).axes[7].set_position([0.443223, 0.245162, 0.029713, 0.135560])
    plt.figure(1).axes[7].get_yaxis().get_label().set_text("# LIF")
    plt.figure(1).text(0.5, 0.5, f'{dataset_name}', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
    plt.figure(1).texts[0].set_position([0.440747, 0.922962])
    plt.figure(1).text(0.5, 0.5, 'b', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[1].new
    plt.figure(1).texts[1].set_position([0.272372, 0.793170])
    plt.figure(1).text(0.5, 0.5, 'a', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[2].new
    plt.figure(1).texts[2].set_position([0.029713, 0.793170])
    plt.figure(1).text(0.5, 0.5, 'constant', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[3].new
    plt.figure(1).texts[3].set_position([0.227802, 0.865276])
    plt.figure(1).text(0.5, 0.5, 'binned uniform', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[4].new
    plt.figure(1).texts[4].set_position([0.703214, 0.865248])
    plt.figure(1).text(0.5, 0.5, 'c', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[5].new
    plt.figure(1).texts[5].set_position([0.539791, 0.793170])
    plt.figure(1).text(0.5, 0.5, 'd', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[6].new
    plt.figure(1).texts[6].set_position([0.775021, 0.793024])
    plt.figure(1).text(0.5, 0.5, 'decay distrib', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[7].new
    plt.figure(1).texts[7].set_position([0.871048, 0.389668])
    plt.figure(1).text(0.5, 0.5, 'decay distrib', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[8].new
    plt.figure(1).texts[8].set_position([0.363680, 0.389668])
    #% end: automatic generated code from pylustrator
    #plt.show()

    if not os.path.exists("./results/figures"):
        os.makedirs("./results/figures")

    plt.savefig(f"./results/figures/{dataset}_measure_vs_decay.pdf")
    plt.close("all")


if __name__ == "__main__":
    num_folds = 5
    simulation_time = 0.5
    dt = 0.005
    first_timestep_to_consider = int((simulation_time * 0.5) / dt)
    folder = f"dt{dt}"
    

    io_or_tts = "io"
    as_histo = False
    bin_no = 25


    experiment = "no_bias_gaussian_weights_128"
    decays = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]


    distribution_range = f" [15, {decays[-1]}]ms"
    for dataset in ["mnist", "fashion_mnist", "cifar10", "letter_mnist"]:

        if dataset == "mnist":
            dataset_name = "        MNIST"
            io_lim = 14.0
            tts_lim = 125.0
        if dataset == "mnist_with_brightness_offset":
            dataset_name = "MNIST adj."
        if dataset == "fashion_mnist":
            dataset_name = "Fashion-MNIST"
            io_lim = 20.0
            tts_lim = 80.0
        if dataset == "cifar10":
            dataset_name = "      CIFAR-10"
            io_lim = 30.0
            tts_lim = 20.0
        if dataset == "letter_mnist":
            dataset_name = "EMNIST/Letters"
            io_lim = 12.0
            tts_lim = 75.0

        io_vs_tdecay()
