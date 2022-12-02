import pylustrator
pylustrator.start()

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

datasets = ["mnist", "fashion_mnist", "letter_mnist", "mnist_with_brightness_offset", "fashion_mnist_with_brightness_offset", "cifar10"]


plt_idx = 1
color_pairs = [('purple', 'brown'), ('olive', '#1f77b4'), ('red', 'orange'), ('green', 'blue')]
for dataset in datasets:
    plt.subplot(2, 3, plt_idx)
    plt_idx += 1

    if dataset == "mnist":
        dataset_name = "MNIST"
        experiment = "no_bias_gaussian_weights_128"

    if dataset == "mnist_with_brightness_offset":
        dataset_name = "MNIST adj."
        experiment = "no_bias_gaussian_weights_128_adj"

    if dataset == "fashion_mnist":
        dataset_name = "Fashion-MNIST"
        experiment = "no_bias_gaussian_weights_128"

    if dataset == "fashion_mnist_with_brightness_offset":
        dataset_name = "Fashion-MNIST adj."
        experiment = "no_bias_gaussian_weights_128_adj"

    if dataset == "cifar10":
        dataset_name = "CIFAR-10"
        experiment = "no_bias_gaussian_weights_128"

    if dataset == "letter_mnist":
        dataset_name = "Letter-MNIST"
        experiment = "no_bias_gaussian_weights_128"


    df_asc = pd.read_csv(f"./results/{dataset}/{experiment}/binned_uniform_intervals16_descending=False.csv")
    df_desc = pd.read_csv(f"./results/{dataset}/{experiment}/binned_uniform_intervals16_descending=True.csv")

    if dataset == "mnist":
        max_acc = 0.9
    elif dataset == "mnist_with_brightness_offset":
        max_acc = 0.8
    elif dataset == "fashion_mnist":
        max_acc = 0.8
    elif dataset == "fashion_mnist_with_brightness_offset":
        max_acc = 0.8
    elif dataset == "cifar10":
        max_acc = 0.4
    elif dataset == "letter_mnist":
        max_acc = 0.7
    elif dataset == "letter_mnist_with_brightness_offset":
        max_acc = 0.7
    else:
        raise ValueError("invalid name for dataset")


    indices = []
    asc_means = []
    asc_stds = []
    desc_means = []
    desc_stds = []
    for i in range(df_asc.shape[0]):
        idx = int(df_asc.iloc[i]["Unnamed: 0"].split("to_")[1])
        
        indices.append(idx)
        asc_means.append(df_asc.iloc[i]["mean_acc"])
        asc_stds.append(df_asc.iloc[i]["std_acc"])
        desc_means.append(df_desc.iloc[i]["mean_acc"])
        desc_stds.append(df_desc.iloc[i]["std_acc"])

    if dataset.__contains__("cifar"):
        color_pair = color_pairs[0]
    elif dataset.__contains__("fashion"):
        color_pair = color_pairs[2]
    elif dataset.__contains__("letter"):
        color_pair = color_pairs[1]
    else:
        color_pair = color_pairs[3]

    plt.scatter(indices, asc_means, color=color_pair[0], alpha=0.8, label="asc.")  # dropout starting from fastest decay
    plt.plot(indices, asc_means, color=color_pair[0], alpha=0.8)
    plt.fill_between(indices, np.asarray(asc_means) - np.asarray(asc_stds), np.asarray(asc_means) + np.asarray(asc_stds), color=color_pair[0], alpha=0.2)
    plt.scatter(indices, desc_means, color=color_pair[1], alpha=0.8, label="desc.")  # dropout starting from slowest decay
    plt.plot(indices, desc_means, color=color_pair[1], alpha=0.8)
    plt.fill_between(indices, np.asarray(desc_means) - np.asarray(desc_stds), np.asarray(desc_means) + np.asarray(desc_stds), color=color_pair[1], alpha=0.2)
    plt.grid(True)
    
    plt.xlabel("ablation [%]")
    plt.ylabel("mean accuracy")
    #plt.tick_params(axis='both', which='both', direction='in',
    #                    labelbottom=True, labeltop=False, labelleft=True, labelright=False,
    #                    bottom=True, top=True, left=True, right=True)
    #plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    #plt.xlim(0, 100)
    plt.ylim(0, max_acc)



#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(21.000000/2.54, 13.900000/2.54, forward=True)
plt.figure(1).axes[0].legend()
plt.figure(1).axes[0].set_position([0.073878, 0.575796, 0.253570, 0.345478])
plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("accuracy")
plt.figure(1).axes[1].set_ylim(0.0, 0.9)
plt.figure(1).axes[1].legend()
plt.figure(1).axes[1].set_position([0.397037, 0.575796, 0.253570, 0.345478])
plt.figure(1).axes[1].get_xaxis().get_label().set_text("")
plt.figure(1).axes[1].get_yaxis().get_label().set_text(" ")
plt.figure(1).axes[2].set_ylim(0.0, 0.9)
plt.figure(1).axes[2].legend()
plt.figure(1).axes[2].set_position([0.730682, 0.575796, 0.253570, 0.345478])
plt.figure(1).axes[2].get_xaxis().get_label().set_text("")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("")
plt.figure(1).axes[3].set_ylim(0.0, 0.9)
plt.figure(1).axes[3].set_position([0.073878, 0.079172, 0.253570, 0.345478])
plt.figure(1).axes[3].get_yaxis().get_label().set_text("accuracy")
plt.figure(1).axes[4].set_ylim(0.0, 0.9)
plt.figure(1).axes[4].set_position([0.397037, 0.079172, 0.253570, 0.345478])
plt.figure(1).axes[4].get_yaxis().get_label().set_text("")
plt.figure(1).axes[5].set_ylim(0.0, 0.45)
plt.figure(1).axes[5].legend()
plt.figure(1).axes[5].set_position([0.730682, 0.079172, 0.253570, 0.345478])
plt.figure(1).axes[5].get_yaxis().get_label().set_text("")
plt.figure(1).text(0.5, 0.5, ' ', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.335415, 0.950064])
plt.figure(1).text(0.5, 0.5, '        MNIST', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.130598, 0.935669])
plt.figure(1).text(0.5, 0.5, 'Fashion-MNIST', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.463766, 0.935669])
plt.figure(1).text(0.5, 0.5, 'EMNIST/Letters', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.797411, 0.935669])
plt.figure(1).text(0.5, 0.5, 'd', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.036701, 0.935669])
plt.figure(1).text(0.5, 0.5, 'f', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.364149, 0.935669])
plt.figure(1).text(0.5, 0.5, 'h', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.698747, 0.935669])
plt.figure(1).text(0.5, 0.5, 'MNIST adj.', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_position([0.153953, 0.442643])
plt.figure(1).text(0.5, 0.5, 'Fashion-MNIST adj.', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[8].new
plt.figure(1).texts[8].set_position([0.451850, 0.442643])
plt.figure(1).text(0.5, 0.5, 'e', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[9].new
plt.figure(1).texts[9].set_position([0.036701, 0.442643])
plt.figure(1).text(0.5, 0.5, 'g', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[10].new
plt.figure(1).texts[10].set_position([0.364149, 0.442643])
plt.figure(1).text(0.5, 0.5, '      CIFAR-10', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[11].new
plt.figure(1).texts[11].set_position([0.785972, 0.442643])
plt.figure(1).text(0.5, 0.5, 'i', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[12].new
plt.figure(1).texts[12].set_position([0.698747, 0.442643])
#% end: automatic generated code from pylustrator
#plt.show()

if not os.path.exists("./results/figures"):
    os.makedirs("./results/figures")

plt.savefig(f"./results/figures/ablation.pdf")
plt.close("all")
