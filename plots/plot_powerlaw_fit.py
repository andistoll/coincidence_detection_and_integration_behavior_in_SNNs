from cProfile import label
import pylustrator
pylustrator.start()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



initializers = ["const", "binned"]


datasets = ["mnist", "mnist_with_brightness_offset", "fashion_mnist", "fashion_mnist_with_brightness_offset", "letter_mnist", "letter_mnist_with_brightness_offset", "cifar10"]
colors = ['#1f77b4', '#17becf', '#9467bd', '#e377c2', 'darkgoldenrod', '#7f7f7f', '#2ca02c']


plt_idx = 1
for i in ["loglog", "normal"]:
    for initializer in initializers:
        plt.subplot(1, 4, plt_idx)
        plt_idx += 1
        color_idx = 0
        for dataset in datasets:
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
            if dataset == "letter_mnist_with_brightness_offset":
                dataset_name = "EMNIST/Letters adj."

            if dataset.__contains__("with_brightness_offset"):
                df = pd.read_csv(f"./results/{dataset}/no_bias_gaussian_weights_128_adj/line_fits_{initializer}.csv")
            else:
                df = pd.read_csv(f"./results/{dataset}/no_bias_gaussian_weights_128/line_fits_{initializer}.csv")
            X = df['decays']
            y = df['slopes']

            def func_powerlaw(x, m, c, c0):
                return 1 - x**m * c + c0

            popt, pcov = curve_fit(func_powerlaw, X, y, p0 = np.asarray([-0.99, -4.42, 0.47]))

            z = np.polyfit(X, y, deg=10)
            print(z)

            p = np.poly1d(z)

            print(popt)
            m, c, c0 = popt[0], popt[1], popt[2]


            plt.plot(X, func_powerlaw(X, m, c, c0) - c0 - 1, '--', label=dataset_name, c=colors[color_idx])

            if i == "normal":
                plt.xticks([0, 400], fontsize=6)
                plt.yticks([0, 0.25], fontsize=6)

            color_idx += 1

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(18.210000/2.54, 6.950000/2.54, forward=True)
plt.figure(1).axes[0].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
plt.figure(1).axes[0].loglog()
plt.figure(1).axes[0].set_position([0.090727, 0.165845, 0.285928, 0.663379])
plt.figure(1).axes[0].get_xaxis().get_label().set_text("decay time (ms)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("slopes")
plt.figure(1).axes[1].set_xticks([0, 100, 200, 300, 400], ["0", "", "200", "", "400"])
plt.figure(1).axes[1].legend(bbox_to_anchor=(1,0), loc="lower left")
plt.figure(1).axes[1].loglog()
plt.figure(1).axes[1].set_position([0.442638, 0.166054, 0.285928, 0.663379])
plt.figure(1).axes[1].get_xaxis().get_label().set_text("decay time (ms)")
plt.figure(1).axes[2].set_ylim(-0.01, 0.32)
plt.figure(1).axes[2].set_position([0.131417, 0.233919, 0.104474, 0.216592])
plt.figure(1).axes[2].spines.right.set_visible(False)
plt.figure(1).axes[2].spines.top.set_visible(False)
plt.figure(1).axes[3].set_ylim(-0.01, 0.32)
plt.figure(1).axes[3].set_position([0.484978, 0.233919, 0.104474, 0.216592])
plt.figure(1).axes[3].spines.right.set_visible(False)
plt.figure(1).axes[3].spines.top.set_visible(False)
plt.figure(1).text(0.5, 0.5, ' ', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.329917, 0.930172])
plt.figure(1).text(0.5, 0.5, 'binned uniform', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.516870, 0.850855])
plt.figure(1).text(0.5, 0.5, 'f', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.406897, 0.850855])
plt.figure(1).text(0.5, 0.5, 'e', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.041240, 0.850855])
plt.figure(1).text(0.5, 0.5, 'constant', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.189702, 0.850855])
#% end: automatic generated code from pylustrator
#plt.show()

if not os.path.exists("./results/figures"):
    os.makedirs("./results/figures")

plt.savefig(f"./results/figures/powerlaw_fits.pdf")
plt.close("all")
