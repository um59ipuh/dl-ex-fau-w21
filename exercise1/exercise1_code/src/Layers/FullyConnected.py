import sys
import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):

    prev_weight_tensor = None

    def __init__(self, input_size, output_size):
        BaseLayer.__init__(self)
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = None

        # generate random weights for synapse; raw + 1 for bias
        self.weights = np.random.rand(self.input_size, self.output_size)

        # Optimizer is not set yet
        self._optimizer = None
        self.input_tensor = None
        self.gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        bias = np.ones((1, input_tensor.shape[1]))
        input_tensor_with_bias = np.concatenate((input_tensor, bias), axis=0)
        # input_tensor_with_bias
        self.tensor_value = np.dot(input_tensor, self.weights)

        print(self.tensor_value.shape)

        return self.tensor_value

    def backward(self, error_tensor):
        gradient = np.dot(self.input_tensor.transpose(), error_tensor)
        if FullyConnected.prev_weight_tensor is not None:
            gradient = np.dot(FullyConnected.prev_weight_tensor.transpose() ,gradient)
        else:
            pass
        if self.optimizer is not None:
            sgd = self.optimizer
            self.weights = sgd.calculate_update(self.weights, gradient)

        self.gradient_weights = gradient

        # save the previous weight
        FullyConnected.prev_weight_tensor = self.weights

        return error_tensor
