import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        BaseLayer.__init__(self)
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(self.input_size, self.output_size) # generates value [0, 1)

        # set the default optimizer as SGD
        self._optimizer = 'sgd'

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        return np.dot(input_tensor, self.weights)

    def backward(self, error_tensor):
        return np.dot(self.optimizer.transpose(), error_tensor)