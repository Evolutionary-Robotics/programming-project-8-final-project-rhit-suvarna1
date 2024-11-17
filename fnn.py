##################################################################################
# Feed-forward neural network for teaching and learning.
# Works with any number of hidden layers/neurons.
# Supports the following activation functions: sigmoidm, tanh, relu, linear, gaussian, and identity.
#
# Eduardo Izquierdo
# September 2024
##################################################################################

import numpy as np

class FNN:
    def __init__(self, units_per_layer):
        """ Create Feedforward Neural Network based on specifications
        units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
        """
        self.units_per_layer = units_per_layer
        self.num_layers = len(units_per_layer)

        # lambdas for supported activation functions
        self.activation = lambda x: 1 / (1 + np.exp(-x))

        self.weightrange = 5
        self.biasrange = 5

    def setParams(self, params):
        """ Set the weights, biases, and activation functions of the neural network 
        Weights and biases are set directly by a parameter;
        The activation function for each layer is set by the parameter with the highest value (one for each possible one out of the six)
        """
        self.weights = []
        start = 0
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l]*self.units_per_layer[l+1]
            self.weights.append((params[start:end]*self.weightrange).reshape(self.units_per_layer[l],self.units_per_layer[l+1]))
            start = end
        self.biases = []
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l+1]
            self.biases.append((params[start:end]*self.biasrange).reshape(1,self.units_per_layer[l+1]))
            start = end

    def forward(self, inputs):
        """ Forward propagate the given inputs through the network """
        states = np.asarray(inputs)
        for l in np.arange(self.num_layers - 1):
            if states.ndim == 1:
                states = [states]
            states = self.activation(np.matmul(states, self.weights[l]) + self.biases[l])
        return states

