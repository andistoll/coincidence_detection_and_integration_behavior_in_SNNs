from warnings import WarningMessage
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.layers import Layer, TimeDistributed, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers

import tensorflow as tf
from tf_spiking.surrogates import get, theta_forget
from tf_spiking.custom_initializers import BinnedUniformDecayInitializer


# define the lif
@tf.function
def lif_gradient(x, w_i, w_l, theta_one, t_thresh=1, dt=1):
    time_steps = x.shape[1]

    Vm = w_i * x[:, 0] * 0
    states = tf.TensorArray(tf.float32, size=time_steps)

    for i in tf.range(time_steps):
        Vm = tf.nn.relu(w_i * x[:, i] + (1 - w_l * dt) * Vm * theta_forget(t_thresh - Vm))
        spike = theta_one(Vm - t_thresh)
        states = states.write(i, spike)

    return tf.transpose(states.stack(), (1, 0, 2))


# define the lif
@tf.function
def lif_sum_no_spike(x, w_i, w_l, dt=1):
    time_steps = x.shape[1]

    Vm = w_i * x[:, 0] * 0
    states = tf.TensorArray(tf.float32, size=time_steps)

    for i in tf.range(time_steps):
        Vm = tf.nn.relu(w_i * x[:, i] + (1 - w_l * dt) * Vm)
        states = states.write(i, Vm)

    return tf.transpose(states.stack(), (1, 0, 2))


# define the lif
@tf.function
def lif_gradient_membrane_potential(x, w_i, w_l, theta_one, t_thresh=1, dt=1):
    time_steps = x.shape[1]

    Vm = w_i * x[:, 0] * 0
    states = tf.TensorArray(tf.float32, size=time_steps)

    for i in tf.range(time_steps):
        Vm = tf.nn.relu(w_i * x[:, i] + (1 - w_l * dt) * Vm * theta_forget(t_thresh - Vm))
        spike = theta_one(Vm - t_thresh)
        states = states.write(i, Vm)

    return tf.transpose(states.stack(), (1, 0, 2))



class LIF_Activation(Layer):

    def __init__(self, units=1, t_thresh=1, surrogate="flat", beta=10, return_potential=False,
                 decay_initializer='random_uniform', dt=1,
                 trainable_leak=True,               # Added option to toggle w_leak trainability
                 no_spike=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.thresh = t_thresh
        self.no_spike = no_spike

        self.dt = dt
        self.trainable_leak = trainable_leak

        self.theta = get(surrogate, beta)
        self.beta = beta                            # beta is irrelevant for flat surrogates
        self.surrogate = surrogate
        self.return_potential = return_potential

        self.decay_initializer = initializers.get(decay_initializer)

    def build(self, input_shape):
        self.w_leak = self.add_weight(
            name='w_leak',
            shape=[self.units],
            initializer=self.decay_initializer,
            trainable=self.trainable_leak   # Added option to toggle w_leak trainability
        )

        super().build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        return {
            "units": self.units,
            "decay_initializer": tf.keras.initializers.serialize(self.decay_initializer),
            "t_thresh": self.thresh,
            "surrogate": self.surrogate,
            "beta": self.beta,
            "no_spike": self.no_spike,
        }

    def call(self, x):
        time_steps = x.shape[1]
        pre_channels = x.shape[2]

        if self.no_spike:
            y = lif_sum_no_spike(x, 1, self.w_leak, self.dt)
            return y

        # integrate (batch, time_steps, pre_channels)
        if self.return_potential:
            y = lif_gradient_membrane_potential(x, 1, self.w_leak, theta_one=self.theta, t_thresh=self.thresh, dt=self.dt)
        else:
            y = lif_gradient(x, 1, self.w_leak, theta_one=self.theta, t_thresh=self.thresh, dt=self.dt)

        # reshape to (batch, time_steps, channels)
        y = tf.reshape(y, (-1, time_steps, pre_channels))
        return y


class SumEnd(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = x[:, -10:, :]
        x = tf.expand_dims(tf.math.reduce_sum(x, axis=1), axis=1)
        x = x / (tf.expand_dims(tf.math.reduce_sum(x, axis=2), axis=2) + 0.001)
        return x[:, 0, :]


class GetEnd(Layer):
    def call(self, x):
        return x[:, -1, :]


class DenseLIF(Sequential):
    def __init__(self, units, gaussian_weights=False, only_pos_weights=False, use_bias=True, dt=1, surrogate="flat", beta=10, return_potential=False, name=None, decay_distribution='constant', trainable_leak=False, t_decay=0.2):

        self.dt = dt
        self.surrogate = surrogate
        self.beta = beta
        self.return_potential = return_potential
        self.units = units
        self.gaussian_weights = gaussian_weights
        self.only_pos_weights = only_pos_weights
        self.use_bias = use_bias


        if self.gaussian_weights:
            kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.12)
        else:
            kernel_initializer = "glorot_uniform"  # Keras default

        if self.only_pos_weights:
            kernel_constraint = tf.keras.constraints.NonNeg()
        else:
            kernel_constraint = None

        # TODO: configure manually for different range
        fastest_t_decay = 0.015 # [s] -- the shortest decay time is restricted to 10ms
        longest_t_decay = 0.480 # [s] -- the longest/slowest decay is restricted to 500ms
        bin_step_size = None # default is 15ms

        if t_decay == fastest_t_decay and decay_distribution.__contains__('upto_fastest'):
            raise Warning(f"Dense_LIF.__init__(): t_decay == fastest_t_decay ({fastest_t_decay}s) results in unexpected behavior.")
        if t_decay == longest_t_decay and (decay_distribution.__contains__('downto_longest') or decay_distribution == 'random_uniform'):
            raise Warning(f"Dense_LIF.__init__(): t_decay == longest_t_decay ({longest_t_decay}s) results in unexpected behavior.")


        if decay_distribution == 'constant':
            decay_initializer = tf.keras.initializers.Constant(1 / t_decay)
        elif decay_distribution.startswith('binned_uniform'):
            decay_initializer = BinnedUniformDecayInitializer(distribution_type=decay_distribution, bin_step_size=bin_step_size, t_decay=t_decay, fastest_t_decay=fastest_t_decay, longest_t_decay=longest_t_decay)
        else:
            raise ValueError("DenseLIF.__init__(): Invalid parameter for decay_distribution.")


        super().__init__([
            TimeDistributed(Dense(self.units, kernel_initializer=kernel_initializer, kernel_constraint=kernel_constraint, use_bias=self.use_bias)),
            LIF_Activation(
                units=self.units,
                decay_initializer=decay_initializer,
                dt=self.dt,
                trainable_leak=trainable_leak,
                surrogate=self.surrogate,
                beta=self.beta, 
                return_potential=self.return_potential),
        ], name=name)

    def get_config(self):
        return {"units": self.units, "dt": self.dt, "name": self.name, "surrogate": self.surrogate, "beta": self.beta,
                "return_potential": self.return_potential, "gaussian_weights": self.gaussian_weights, "use_bias": self.use_bias}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DenseLIFNoSpike(Sequential):
    def __init__(self, units=10, dt=1, name=None):
        super().__init__([
            TimeDistributed(Dense(units)),
            LIF_Activation(
                decay_initializer=tf.keras.initializers.Constant(0),
                no_spike=True,
                dt=dt
            ),
            GetEnd(),
            tf.keras.layers.Softmax(),
        ], name=name)
        self.units = units

    def get_config(self):
        return {"units": self.units, "name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
