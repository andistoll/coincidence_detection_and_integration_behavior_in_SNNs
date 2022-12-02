import pylustrator
pylustrator.start()
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


experiment = "no_bias_gaussian_weights_128"

# Figure 3a, b
datasets1 = ['mnist', 'fashion_mnist', 'letter_mnist', 'cifar10']
datasets1_color = ['#1f77b4', '#9467bd', 'darkgoldenrod', '#2ca02c']

# Figure 3c, d
datasets2 = ['mnist', 'mnist_with_brightness_offset', 'fashion_mnist', 'fashion_mnist_with_brightness_offset', 'letter_mnist', 'letter_mnist_with_brightness_offset']
datasets2_color = ['#1f77b4', '#17becf', '#9467bd', '#e377c2', 'darkgoldenrod', '#7f7f7f']

initializers = ['const', 'binned']

plt_idx = 1
for datasets in [datasets1, datasets2]:
    for initializer in initializers:
        plt.subplot(1, 4, plt_idx)
        plt_idx += 1

        color_idx = 0
        for dataset in datasets:
            if dataset.__contains__("with_brightness_offset"):
                df = pd.read_csv(f"./results/{dataset}/no_bias_gaussian_weights_128_adj/line_fits_{initializer}.csv")
            else:
                df = pd.read_csv(f"./results/{dataset}/{experiment}/line_fits_{initializer}.csv")

            if dataset == "mnist":
                dataset_name = "MNIST"
            if dataset == "mnist_with_brightness_offset":
                dataset_name = "MNIST adj."
            if dataset == "fashion_mnist":
                dataset_name = "Fashion-MNIST"
            if dataset == "fashion_mnist_with_brightness_offset":
                dataset_name = "Fashion-MNIST adj."
            if dataset == "cifar10":
                dataset_name = "CIFAR-10"
            if dataset == "letter_mnist":
                dataset_name = "EMNIST/Letters"
            if dataset == "mnist_wbo_letter":
                dataset_name = "MNIST adj. to letter"
            if dataset == "letter_mnist_with_brightness_offset":
                dataset_name = "EMNIST/Letters adj."


            s=5
            if datasets.__contains__("mnist_with_brightness_offset"):
                plt.scatter(df["decays"], df["slopes"], c=datasets2_color[color_idx], s=s)
                plt.plot(df["decays"], df["slopes"], label=f"{dataset_name}", c=datasets2_color[color_idx])
                color_idx += 1
            else:
                plt.scatter(df["decays"], df["slopes"], c=datasets1_color[color_idx], s=s)
                plt.plot(df["decays"], df["slopes"], label=f"{dataset_name}", c=datasets1_color[color_idx])
                color_idx += 1
            #plt.tick_params(axis='both', which='both', direction='in',
            #            labelbottom=True, labeltop=False, labelleft=True, labelright=False,
            #            bottom=True, top=True, left=True, right=True)


        plt.xlabel("decay time (ms)")
        plt.ylabel("slopes")

        if experiment == "no_bias_gaussian_weights_128_decay120":
            plt.xlim(0, 135)
        else:
            plt.xlim(0, 500)


        if dataset.__contains__('mnist_with_brightness_offset'):
            plt.ylim(0, 0.6)
            plt.yticks(np.arange(0, 0.7, 0.1), ["0.0", "", "0.2", "", "0.4", "", "0.6"])
        else:
            plt.ylim(0, 1.9)
            plt.yticks(np.arange(0, 1.9, 0.1), ["0.0", "", "0.2", "", "0.4", "", "0.6", "", "0.8" ,"", "1.0", "", "1.2", "", "1.4", "", "1.6", "", "1.8"])

    
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(18.210000/2.54, 13.900000/2.54, forward=True)
plt.figure(1).axes[0].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
plt.figure(1).axes[0].set_position([0.076981, 0.568599, 0.291427, 0.338280])
plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
plt.figure(1).axes[1].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
plt.figure(1).axes[1].legend(bbox_to_anchor=(1,0), loc="lower left")
plt.figure(1).axes[1].set_position([0.437140, 0.568599, 0.291427, 0.338280])
plt.figure(1).axes[1].get_xaxis().get_label().set_text("")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("")
plt.figure(1).axes[2].set_ylim(0.0, 0.6500000000000001)
plt.figure(1).axes[2].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
plt.figure(1).axes[2].set_position([0.076981, 0.100764, 0.291427, 0.338280])
plt.figure(1).axes[2].get_yaxis().get_label().set_text("slopes")
plt.figure(1).axes[3].set_ylim(0.0, 0.6500000000000001)
plt.figure(1).axes[3].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
plt.figure(1).axes[3].legend(bbox_to_anchor=(1,0), loc="lower left")
plt.figure(1).axes[3].set_position([0.437140, 0.100764, 0.291427, 0.338280])
plt.figure(1).axes[3].get_yaxis().get_label().set_text("")
plt.figure(1).text(0.5, 0.5, f' ', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.131029, 0.935039])
plt.figure(1).text(0.5, 0.5, 'constant', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.175956, 0.917676])
plt.figure(1).text(0.5, 0.5, 'binned uniform', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.519619, 0.917676])
plt.figure(1).text(0.5, 0.5, 'c', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.016496, 0.453440])
plt.figure(1).text(0.5, 0.5, 'a', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.016496, 0.917676])
plt.figure(1).text(0.5, 0.5, 'constant', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.175956, 0.453440])
plt.figure(1).text(0.5, 0.5, 'binned uniform', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.519619, 0.453440])
plt.figure(1).text(0.5, 0.5, 'b', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_position([0.401399, 0.917676])
plt.figure(1).text(0.5, 0.5, 'd', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[8].new
plt.figure(1).texts[8].set_position([0.401399, 0.453440])
#% end: automatic generated code from pylustrator
#plt.show()

if not os.path.exists("./results/figures"):
    os.makedirs("./results/figures")

plt.savefig(f"./results/figures/slopes.pdf")
plt.close("all")
