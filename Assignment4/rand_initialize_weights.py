import numpy as np


def rand_initialize_weights(input_layer, output_layer, epsilon):

    return np.random.rand(output_layer, input_layer+1)*(2*epsilon) - epsilon

