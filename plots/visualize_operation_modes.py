import pylustrator
pylustrator.start()
import os
import tensorflow.keras as keras
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

plt_idx = 2
for dec in [0.48, 0.03]:
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
    model.layers[1].set_weights([model.layers[1].get_weights()[0] * 0 + 0.4, model.layers[1].get_weights()[1], model.layers[1].get_weights()[2]])

    print(model.layers[1].get_weights())


    rndm_draws = [10, 10, 10, 10, 10, 10, 10, 100, 40, 40, 40, 40, 40, 30, 30, 15, 15, 10, 10, 10, 10, 15] 
    # Create some input spike train
    x = np.zeros((1, time_steps, 784))
    t = 0
    j = 0

    idx = 0
    for i in range(1, time_steps - 1):
        t -= int(1/dt) # sampling frequency
        if t <= 0:
            t_between_spikes = rndm_draws[idx]  # [ms]
            t = int(t_between_spikes / 1000 / dt)
            idx += 1
        j += t

        try:
            x[0, int(j), int(j)] = 1
        except IndexError:
            break


    get_lif_layer_output = keras.backend.function([model.layers[1].input], [model.layers[1].output])

    lif_img = get_lif_layer_output([x])[0]
    lif_img = deepcopy(lif_img[0])

    xs = np.arange(time_steps) * dt * 1000 


    plt.subplot(2, 3, plt_idx)
    plt_idx += 1
    plt.plot(xs, lif_img[:, 0])
    plt.plot([-20, 520], [1, 1], color='gray', linestyle='dashed', alpha=0.5)

    plt.xlabel("time [ms]")
    plt.ylabel("Vm")
    plt.ylim((0, 1.2))
    # additionally mark the spikes with red dots
    spike_times = np.where(lif_img[:, 0] >= 1)[0]  # [ms]

    for s in spike_times:
        timestep_of_spike = int(s * 1000 * dt) # convert spike_times [ms] to time_steps


    plt.subplot(2, 3, plt_idx)
    plt_idx += 2
    plt.plot(xs, np.zeros_like(xs), color='#1f77b4')
    spike_times = np.where(lif_img[:, 0] >= 1)[0]  # [ms]
    for s in spike_times:
        timestep_of_spike = int(s * 1000 * dt) # convert spike_times [ms] to time_steps
        #plt.plot(timestep_of_spike, lif_img[:, 0][s], 'ro')
        plt.plot([timestep_of_spike, timestep_of_spike],[0, 1], color='#1f77b4')

    plt.xlabel("time [ms]")
    plt.ylabel("spikes")
    plt.ylim((0, 1.2))

    


#################################
# Input spike train:

idx = 0
ax0 = plt.subplot(2, 3, 1)
ax0.plot(xs, np.zeros_like(xs))
ax0.set_ylabel("spikes")
ax0.set_ylim(0, 1.2)
# additionally mark the spikes with red dots
spike_times = np.where(x[0] >= 1)[0]  # [ms]
for s in spike_times:
    timestep_of_spike = int(s * 1000 * dt) # convert spike_times [ms] to time_steps
    ax0.plot([timestep_of_spike, timestep_of_spike],[0, 1])



#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(18.210000/2.54, 6.950000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(-20.0, 520.0)
plt.figure(1).axes[0].set_ylim(0.0, 1.25)
plt.figure(1).axes[0].set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
plt.figure(1).axes[0].set_position([0.384903, 0.605694, 0.241939, 0.288425])
plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
plt.figure(1).axes[1].set_xlim(-20.0, 520.0)
plt.figure(1).axes[1].set_ylim(0.0, 1.25)
plt.figure(1).axes[1].set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
plt.figure(1).axes[1].set_position([0.731316, 0.605694, 0.241939, 0.288425])
plt.figure(1).axes[1].get_xaxis().get_label().set_text("")
plt.figure(1).axes[2].set_xlim(-20.0, 520.0)
plt.figure(1).axes[2].set_ylim(0.0, 1.25)
plt.figure(1).axes[2].set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
plt.figure(1).axes[2].set_position([0.384903, 0.165845, 0.241939, 0.288425])
plt.figure(1).axes[2].get_xaxis().get_label().set_text("simulation time (ms)")
plt.figure(1).axes[3].set_xlim(-20.0, 520.0)
plt.figure(1).axes[3].set_ylim(0.0, 1.25)
plt.figure(1).axes[3].set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
plt.figure(1).axes[3].set_position([0.731316, 0.165845, 0.241939, 0.288425])
plt.figure(1).axes[3].get_xaxis().get_label().set_text("simulation time (ms)")
plt.figure(1).axes[4].set_xlim(-20.0, 520.0)
plt.figure(1).axes[4].set_ylim(0.0, 1.25)
plt.figure(1).axes[4].set_xticks([0, 100, 200, 300, 400, 500], ["0", "", "200", "", "400", ""])
plt.figure(1).axes[4].set_position([0.057735, 0.366300, 0.241939, 0.288425])
plt.figure(1).axes[4].get_xaxis().get_label().set_text("simulation time (ms)")
plt.figure(1).text(0.5, 0.5, 'input spike train', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.098975, 0.663379])
plt.figure(1).text(0.5, 0.5, 'decay time: 480ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.417895, 0.901330])
plt.figure(1).text(0.5, 0.5, 'decay time: 30ms', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.420644, 0.461481])
plt.figure(1).text(0.5, 0.5, 'output spike train', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[3].new
plt.figure(1).texts[3].set_position([0.767057, 0.461481])
plt.figure(1).text(0.5, 0.5, 'output spike train', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.767057, 0.901330])
plt.figure(1).text(0.5, 0.5, 'c', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.027493, 0.901330])
plt.figure(1).text(0.5, 0.5, 'd', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[6].new
plt.figure(1).texts[6].set_position([0.346413, 0.901330])
plt.figure(1).text(0.5, 0.5, 'e', transform=plt.figure(1).transFigure, weight='bold')  # id=plt.figure(1).texts[7].new
plt.figure(1).texts[7].set_position([0.687327, 0.901330])
#% end: automatic generated code from pylustrator
#plt.show()

if not os.path.exists("./results/figures"):
    os.makedirs("./results/figures")

plt.savefig(f"./results/figures/operation_modes.pdf")
