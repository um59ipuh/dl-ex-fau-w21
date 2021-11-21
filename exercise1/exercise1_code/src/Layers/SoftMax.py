import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        X = input_tensor
        ex = np.exp(X - np.max(X))
        return ex / np.sum(ex, axis=0)

    def backward(self, error_tensor):
        # dot product would be f`(error)
        x = self.input_tensor
        ex = np.exp(x - np.max(x))
        soft_dev = ex / np.sum(ex, axis=0)
        return np.multiply(error_tensor, soft_dev)