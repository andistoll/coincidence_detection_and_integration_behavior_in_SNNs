import pylustrator
pylustrator.start()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

datasets = ["mnist", "fashion_mnist", "letter_mnist"]

experiment = "no_bias_gaussian_weights_128"

decays = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480, 600] # the last one (600) is for plotting "binned uniform"


plt_idx = 1
for dataset in datasets:
    df_const = pd.read_csv(f"./results/{dataset}/{experiment}/metrics_constant.csv")
    df_binned = pd.read_csv(f"./results/{dataset}/{experiment}/metrics_distributions.csv")

    if dataset == "mnist":
        dataset_name = "        MNIST"
    if dataset == "mnist_with_brightness_offset":
        dataset_name = "MNIST adj."
    if dataset == "fashion_mnist":
        dataset_name = "Fashion-MNIST"
    if dataset == "fashion_mnist_with_brightness_offset":
        dataset_name = "Fashion-MNIST adj."
    if dataset == "cifar10":
        dataset_name = "      CIFAR-10"
    if dataset == "letter_mnist":
        dataset_name = "EMNIST/Letters"

    metric = "acc"
    plt.subplot(1, 3, plt_idx)
    plt_idx += 1

    means = list(deepcopy(df_const[f"mean_{metric}"]))
    stds = list(deepcopy(df_const[f"std_{metric}"]))

    means.append(deepcopy(df_binned[f"mean_{metric}"]))
    stds.append(deepcopy(df_binned[f"std_{metric}"]))

    means, stds, decays = np.asarray(means), np.asarray(stds), np.asarray(decays)

    plt.scatter(decays, means)
    plt.grid(True)
    for i in range(len(decays)):
        plt.errorbar(decays[i], means[i], yerr=stds[i], fmt="o", c='#1f77b4')

    ylim = plt.gca().get_ylim()
    plt.xlabel("decay time (ms)")
    plt.plot([540, 540], [0, 1], color='gray', linestyle='dashed')
    plt.xticks(decays, ['15', '', '', '', '', '90', '180', '285', '390', '480', 'binned'])


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(21.000000/2.54, 6.950000/2.54, forward=True)
plt.figure(1).axes[0].set_ylim(0.8275, 0.8625)
plt.figure(1).axes[0].set_position([0.073878, 0.158634, 0.253570, 0.692221])
plt.figure(1).axes[0].get_yaxis().get_label().set_text("accuracy")
plt.figure(1).axes[1].set_ylim(0.735, 0.765)
plt.figure(1).axes[1].set_position([0.397037, 0.158634, 0.253570, 0.692221])
plt.figure(1).axes[2].set_ylim(0.6105, 0.6575)
plt.figure(1).axes[2].set_position([0.730682, 0.158634, 0.253570, 0.692221])
plt.figure(1).text(0.5, 0.5, f'', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.445177, 0.927288])
plt.figure(1).text(0.5, 0.5, '        MNIST', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.133458, 0.875371])
plt.figure(1).text(0.5, 0.5, 'Fashion-MNIST', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.466149, 0.875371])
plt.figure(1).text(0.5, 0.5, 'EMNIST/Letters', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.796934, 0.875371])
plt.figure(1).text(0.5, 0.5, 'a', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.037654, 0.875371])
plt.figure(1).text(0.5, 0.5, 'b', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.357000, 0.875371])
plt.figure(1).text(0.5, 0.5, 'c', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.690645, 0.875371])
#% end: automatic generated code from pylustrator
#plt.show()

if not os.path.exists("./results/figures"):
    os.makedirs("./results/figures")

plt.savefig(f"./results/figures/accuracy.pdf")
plt.close("all")
