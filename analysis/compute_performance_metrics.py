from copy import deepcopy
import os
import numpy as np
import pandas as pd
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


dataset = "mnist" # options: mnist, letter_mnist, fashion_mnist, cifar10 -- mnist_with_brightness_offset, letter_mnist_with_brightness_offset, fashion_mnist_with_brightness_offset
experiment = "no_bias_gaussian_weights_128" # options: no_bias_gaussian_weights_128, no_bias_gaussian_weights_128_adj


for distrib in ['constant', 'binned_uniform']:
	decays = [15, 30, 45, 60, 75, 90, 180, 285, 390, 480]

	# Default configuration - will be automatically reconfigured according to the name of the experiment
	num_folds = 5
	gaussian_weights = False
	use_bias = True
	simulation_time = 0.5
	dt = 0.005
	timesteps = int(simulation_time / dt)
	folder_name = f"dt{dt}"
	num_neurons = 128

	# If this parameter changes you will get an error; the default param value of use_bias in Dense_LIF() (integrate_and_fire.py) has to be changed by hand!
	if experiment.__contains__("no_bias"):
		use_bias = False

	if experiment.__contains__("gaussian_weights"):
		gaussian_weights = True



	# load the dataset
	if dataset == "mnist":
		(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
		min_acc, max_acc = 0.8275, 0.8625
		min_AUC, max_AUC = 0.9, 1.0
	    
	elif dataset == "mnist_with_brightness_offset":
		(x_train, y_train), (x_test, y_test) = load_mnist_with_brightness_offset_for_evaluation()
		min_acc, max_acc = 0.6, 0.8
		min_AUC, max_AUC = 0.85, 0.9
	    
	elif dataset == "letter_mnist":
		(x_train, y_train), (x_test, y_test) = load_local_letter_mnist_for_evaluation()
		min_acc, max_acc = 0.6105, 0.6575
		min_AUC, max_AUC = 0.8, 0.9
	    
	elif dataset == "letter_mnist_with_brightness_offset":
		if experiment.__contains__("neg_offset"):
			(x_train, y_train), (x_test, y_test) = load_letter_mnist_with_negative_brightness_offset_for_evaluation()
		else:
			(x_train, y_train), (x_test, y_test) = load_letter_mnist_with_positive_brightness_offset_for_evaluation()
			min_acc, max_acc = 0.4, 0.65
			min_AUC, max_AUC = 0.7, 0.9
	    
	elif dataset == "fashion_mnist":
		(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
		min_acc, max_acc = 0.735, 0.765
		min_AUC, max_AUC = 0.85, 0.9
	    
	elif dataset == "fashion_mnist_with_brightness_offset":
		(x_train, y_train), (x_test, y_test) = load_fashion_mnist_with_brightness_offset_for_evaluation()
		min_acc, max_acc = 0.6, 0.8
		min_AUC, max_AUC = 0.85, 0.9
	    
	elif dataset == "cifar10":
		(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
		min_acc, max_acc = 0.2, 0.4
		min_AUC, max_AUC = 0.7, 0.8
	else:
		raise NotImplementedError(f"{dataset} needs to be loaded...")

	mean_accs, mean_aucs, mean_f1s = {}, {}, {}
	std_accs, std_aucs, std_f1s = {}, {}, {}


	for t_decay in decays:
		print(f"t_decay={t_decay}")

		# we can skip t_decay > 15 for binned uniform (binned_uniform and t_decay=15ms means binned uniform in [15, 480]ms)
		if distrib != 'constant' and t_decay > 15:
			continue

		accs, f1s, aucs = [], [], []

		for fold in range(num_folds):

			# Setup model and load weights
			# generate the model
			num_lif_units = num_neurons
			if dataset.__contains__("letter_mnist"):
				num_classes = 27
			else:
				num_classes = 10
			t_dec = t_decay / 1000 # decay time of Vm      [ms]

			model = keras.models.Sequential([
			keras.Input(x_test.shape[1:]),
			IntensityToPoissonSpiking(timesteps, 1/255, dt=dt),
			DenseLIF(
				num_lif_units,
				gaussian_weights=gaussian_weights,
				use_bias=use_bias,
				surrogate="flat", 
				dt=dt,
				decay_distribution=distrib,
				trainable_leak=False, 
				t_decay=t_dec),
			DenseLIFNoSpike(num_classes, dt),
			])
			optim = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
			loss = keras.losses.MeanSquaredError()
			metrics = [ 'accuracy',
				'AUC',
				tfa.metrics.F1Score(num_classes=num_classes, average='macro')]

			# compile the model
			model.compile(
			optimizer=optim,
			loss=loss,
			metrics=metrics,
			)

			# Load the trained weights and pass them to the model
			path = f"./models/{folder_name}/{experiment}/{dataset}_time{timesteps}_dt{dt}_poisson_hidden{num_lif_units}_{distrib}_decay{t_decay}ms_fold{fold}/"
			saved_model = keras.models.load_model(path)

			model.set_weights(saved_model.get_weights())


			# Recompute the performance metrics
			test_loss, test_accuracy, test_auc, test_f1 = model.evaluate(x_test, keras.utils.to_categorical(y_test))
			
			accs.append(deepcopy(test_accuracy))
			aucs.append(deepcopy(test_auc))
			f1s.append(deepcopy(test_f1))

			keras.backend.clear_session()


		# Compute the mean and std of the metrics
		mean_accs[f'{distrib}_{t_decay}ms'] = np.mean(accs)
		mean_aucs[f'{distrib}_{t_decay}ms'] = np.mean(aucs)
		mean_f1s[f'{distrib}_{t_decay}ms'] = np.mean(f1s)
		std_accs[f'{distrib}_{t_decay}ms'] = np.std(accs)
		std_aucs[f'{distrib}_{t_decay}ms'] = np.std(aucs)
		std_f1s[f'{distrib}_{t_decay}ms'] = np.std(f1s)


	# If the plots have been created once, we want to delete the csv table and create a new one (to avoid appending new values)
	if os.path.exists(f'./results/{dataset}/{experiment}/metrics.csv'):
		os.remove(f'./results/{dataset}/{experiment}/metrics.csv')

	dict = {
	    "mean_acc": mean_accs,
	    "mean_AUC": mean_aucs,
	    "mean_F1": mean_f1s,
	    "std_acc": std_accs,
	    "std_AUC": std_aucs,
	    "std_F1": std_f1s,
	}


	df = pd.DataFrame.from_dict(dict)

	if distrib == 'constant':
		df.to_csv(f'./results/{dataset}/{experiment}/metrics_constant.csv')
	else:
		df.to_csv(f'./results/{dataset}/{experiment}/metrics_distributions.csv')

