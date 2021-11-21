import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        rel_der = np.maximum(1, self.input_tensor)
        return np.multiply(error_tensor, rel_der)