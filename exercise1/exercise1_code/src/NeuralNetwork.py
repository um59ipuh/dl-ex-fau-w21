import numpy as np
import scipy as sp
import copy
# import all
from Layers import *
from Optimization import *

class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        # input_tensor, label_tensor
        self.x, self.y = self.data_layer.next()

        # iterate through all layers and pass input_tensor to layer by layer and get output untill loss function
        # input value for the first time
        neuron_value = self.x
        for layer in self.layers:
            # perform layer's forward
            neuron_value = layer.forward(neuron_value)
            # save result for passing it to next forward
        # last layers value is predictive value
        y_cap = neuron_value

        self.loss = Loss.CrossEntropyLoss()
        loss_value = self.loss.forward(y_cap, self.y)

        return loss_value

    def backward(self):
        # get error from loss function
        error_value = self.loss.backward(self.y)
        # iterate all layers to back-propagate
        add_len_one = len(self.layers)+1
        for i in range(1, add_len_one):
            layer = self.layers[-i]
            error_value = layer.backward(error_value)
        return error_value

    def append_layer(self, layer):
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            error = self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        # iterate through the network
        neuron_value = input_tensor
        for layer in self.layers:
            neuron_value = layer.forward(neuron_value)
        y_cap = neuron_value
        return y_cap