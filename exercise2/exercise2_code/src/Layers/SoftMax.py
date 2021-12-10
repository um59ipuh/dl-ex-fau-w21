import numpy as np
from .Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.class_input_tensor = []
        self.class_probabilities = []

    def forward(self, input_tensor):
        self.class_input_tensor = input_tensor

        max_ = np.max(input_tensor)
        input_tensor = input_tensor - max_
        exp_input_tensor = np.exp(input_tensor)
        sum_ = np.sum(exp_input_tensor, axis=1)
        probabilities = (exp_input_tensor.T / sum_).T

        self.class_probabilities = probabilities

        return probabilities

    def backward(self, error_tensor):
        multiply = np.multiply(error_tensor, self.class_probabilities)
        sum_ = np.sum(multiply, axis=1)
        subtraction = (error_tensor.T - sum_).T
        updated_error_tensor = np.multiply(self.class_probabilities, subtraction)

        return updated_error_tensor
