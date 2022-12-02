import numpy as np
import tensorflow as tf


class BinnedUniformDecayInitializer(tf.keras.initializers.Initializer):

    def __init__(self, distribution_type, t_decay, bin_step_size=None, fastest_t_decay=0.015, longest_t_decay=0.48, show_distribution=False):
        self.show_distribution = show_distribution
        self.distribution_type = distribution_type
        self.t_decay = t_decay
        self.bin_step_size = bin_step_size
        self.fastest_t_decay = fastest_t_decay
        self.longest_t_decay = longest_t_decay

        if not distribution_type == 'binned_uniform' and not distribution_type == 'binned_uniform_upto_fastest_decay':
            raise ValueError("User code: invalid parameter for 'distribution_type' in BinnedUniformDecayInitializer().")

    def __call__(self, shape, dtype=None):
        
        if self.distribution_type == 'binned_uniform':
            if self.bin_step_size == None:
                number_of_bins = int(self.longest_t_decay / self.t_decay)
            else:
                number_of_bins = int((self.longest_t_decay - self.t_decay) / self.bin_step_size) + 1

        else: # 'binned_uniform_upto_fastest_decay'
            if self.bin_step_size == None:
                number_of_bins = int(self.t_decay / self.fastest_t_decay)
            else:
                number_of_bins = int((self.longest_t_decay - self.t_decay) / self.bin_step_size) + 1

        decays = np.ones(shape).flatten()
        
        #print(number_of_bins)
        number_of_neurons_per_bin = int(decays.size / number_of_bins)

        # prepare the binned array
        c = 1
        for i in range(0, decays.size, number_of_neurons_per_bin):
            decays[i:i+number_of_neurons_per_bin] = c
            c += 1

        # shuffle the array
        np.random.shuffle(decays)

        # set the array values to the binned decay times
        if self.distribution_type == 'binned_uniform':
            if self.bin_step_size == None:
                decays *= self.t_decay
            else:
                decays *= self.bin_step_size
                decays += (self.t_decay - self.bin_step_size)
        else:
            if self.bin_step_size == None:
                decays *= self.fastest_t_decay
            else:
                decays *= self.bin_step_size
                decays += (self.fastest_t_decay - self.bin_step_size)

        #print(np.unique(decays))

        # Convert the decay-time-distribution to a leak-value-distribution
        s = 1 / decays   #   leak ~ 1 / t_decay

        
        if self.show_distribution:
            import matplotlib.pyplot as plt
            # flatten shape in case shape is not 1D
            units = 1
            for i in shape:
                units *= i

            plt.hist(decays * 1000, bins=100)
            plt.title(f"{self.distribution_type}")
            plt.xlabel("decay time [ms]")
            plt.ylabel("count")
            plt.yticks([0, 1, 2, 3, 4])
            plt.xlim(0, 500)
            plt.tick_params(axis='both', which='both', direction='in',
                        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                        bottom=True, top=True, left=True, right=True)
            plt.show()


        return tf.convert_to_tensor(s.reshape(shape), dtype=dtype)

        

    def get_config(self):
        return {'distribution_type': self.distribution_type, 'show_distribution': self.show_distribution}
        


# For testing
if __name__ == "__main__":

    t_decay = 15   # [ms]
    t_decay /= 1000

    #init = BinnedUniformDecayInitializer(distribution_type='binned_uniform', bin_step_size=0.007, longest_t_decay=0.12, t_decay=t_decay, show_distribution=True)
    
    # default bin_step_size: 15ms
    init = BinnedUniformDecayInitializer(distribution_type='binned_uniform', bin_step_size=None, longest_t_decay=0.48, t_decay=t_decay, show_distribution=True)
    print(init.__call__((128, )))
 
