import pylustrator
pylustrator.start()
import os
import tensorflow.keras as keras
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from copy import deepcopy

plt_idx = 1
for dec in [0.09]:
    simulation_time = 0.5   # [s]

    # load the dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # generate the model
    num_lif_units = 128
    num_classes = 10
    dt = 0.005              # [s] -- time resolution
    time_steps = int(simulation_time / dt)

    t_decays = np.array([15, 20, 30, 40, 50, 75, 100, 200, 300, 400, 480]) / 1000  # decay time of the membrane potential [ms] => / 1000 for [s]

    t_decay = dec#t_decays[-1]

    model = keras.models.Sequential([
                keras.Input(x_train.shape[1:]),
                IntensityToPoissonSpiking(time_steps, 1/255, dt=dt),
                DenseLIF(num_lif_units, 
                    surrogate="flat", 
                    dt=dt,
                    return_potential=True,
                    decay_distribution='constant',
                    trainable_leak=False, 
                    t_decay=t_decay),
                DenseLIFNoSpike(num_classes, dt),
            ])

    print(model.summary())

    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error',
                metrics=['accuracy'])

    print(model.layers[1].get_weights())
    model.layers[1].set_weights([model.layers[1].get_weights()[0] * 0 + 0.25, model.layers[1].get_weights()[1], model.layers[1].get_weights()[2]])

    print(model.layers[1].get_weights())


    rndm_draws = [20, 240, 15, 15, 15, 15, 15, 30, 15, 15] 
    # Create some input spike train
    x = np.zeros((1, time_steps, 784))
    t = 0
    j = 0

    idx = 0
    for i in range(1, len(rndm_draws)+1):
        t -= int(1/dt) # sampling frequency
        if t <= 0:
            t_between_spikes = rndm_draws[idx]  # [ms]
            t = int(t_between_spikes / 1000 / dt)
            idx += 1
        j += t

        #print(int(j))
        try:
            x[0, int(j), int(j)] = 1
        except IndexError:
            break


    get_lif_layer_output = keras.backend.function([model.layers[1].input], [model.layers[1].output])

    lif_img = get_lif_layer_output([x])[0]
    lif_img = deepcopy(lif_img[0])

    xs = np.arange(time_steps) * dt * 1000 

    plt.subplot(1, 3, plt_idx)
    plt_idx += 1
    plt.plot(xs, lif_img[:, 0])
    plt.plot([-20, 520], [1, 1], color='gray', linestyle='dashed', alpha=0.5)
    plt.xlabel("simulation time (ms)")
    plt.ylabel("Vm")
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "", "", "", "", "1"])
    plt.ylim((0, 1.2))
    # additionally mark the spikes with red dots

    spike_times = []
    for t in range(lif_img.shape[0]):
        if 30 < t < 70:
            if lif_img[t, 0] > lif_img[t-1, 0]:
                spike_times.append(t)

    for s in spike_times:
        timestep_of_spike = int(s * 1000 * dt) # convert spike_times [ms] to time_steps
        plt.plot(timestep_of_spike, lif_img[:, 0][s], 'ro', markersize=3)
        

    plt.subplot(1, 3, plt_idx)
    plt_idx += 1
    plt.plot(xs, lif_img[:, 0])
    plt.plot([-20, 520], [1, 1], color='gray', linestyle='dashed', alpha=0.5)
    plt.arrow(270, 0.05, 40, 0.0, color='red', shape='full', head_width=0.05, head_length=20)
    plt.arrow(330, 0.05, -45, 0.0, color='red', shape='full', head_width=0.05, head_length=20)
    plt.xlabel("simulation time (ms)")
    plt.ylabel("Vm")
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "", "", "", "", "1"])
    plt.ylim((0, 1.2))


plt.subplot(1, 3, plt_idx)
plt.plot([0, 70], [6, 6], color='red', linestyle='dashed')
plt.plot([75, 75], [0, 5.4], color='red', linestyle='dashed')
plt.scatter(75, 6)
plt.xlim(0, 175)
plt.ylim(0, 20)
plt.imshow(plt.imread("./plots/operation_mode_background.png"), extent=[0, 175, 0, 20], aspect='auto')




#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(18.210000/2.54, 6.950000/2.54, forward=True)
plt.figure(1).axes[0].set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
plt.figure(1).axes[0].set_position([0.074231, 0.223530, 0.220494, 0.620115])
plt.figure(1).axes[1].set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
plt.figure(1).axes[1].set_position([0.404148, 0.223530, 0.220494, 0.620115])
plt.figure(1).axes[2].set_xticks([0, 25, 50, 75, 100, 125, 150], ["0", "", "50", "", "100", "", "150"])
plt.figure(1).axes[2].set_position([0.734065, 0.223530, 0.220494, 0.620115])
plt.figure(1).axes[2].get_xaxis().get_label().set_text("eff. integration \n interval (ms)")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("# contrib. input spikes")
plt.figure(1).text(0.5, 0.5, 'contributing input spikes', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.063234, 0.886908])
plt.figure(1).text(0.5, 0.5, 'effective integration interval', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.374456, 0.886908])
plt.figure(1).text(0.5, 0.5, 'operation mode', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.767057, 0.886908])
plt.figure(1).text(0.5, 0.5, 'Coin.Det.', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.737364, 0.310057])
plt.figure(1).text(0.5, 0.5, 'Integrator', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.849536, 0.692221])
plt.figure(1).text(0.5, 0.5, 'f', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.032992, 0.886908])
plt.figure(1).text(0.5, 0.5, 'g', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.340914, 0.886908])
plt.figure(1).text(0.5, 0.5, 'h', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_position([0.731316, 0.886908])
plt.figure(1).text(0.5, 0.5, '6', transform=plt.figure(1).transFigure, c='red')  # id=plt.figure(1).texts[8].new
plt.figure(1).texts[8].set_position([0.195201, 0.274004])
plt.figure(1).text(0.5, 0.5, '75', transform=plt.figure(1).transFigure, c='red')  # id=plt.figure(1).texts[9].new
plt.figure(1).texts[9].set_position([0.519619, 0.274004])
plt.figure(1).text(0.5, 0.5, '6', transform=plt.figure(1).transFigure, c='red')  # id=plt.figure(1).texts[10].new
plt.figure(1).texts[10].set_position([0.717569, 0.398027])
plt.figure(1).text(0.5, 0.5, '75', transform=plt.figure(1).transFigure, c='red')  # id=plt.figure(1).texts[11].new
plt.figure(1).texts[11].set_position([0.816544, 0.165845])
#% end: automatic generated code from pylustrator
#plt.show()

if not os.path.exists("./results/figures"):
    os.makedirs("./results/figures")

plt.savefig(f"./results/figures/measures.pdf")
