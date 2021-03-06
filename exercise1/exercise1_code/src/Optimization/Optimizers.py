import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # minimize loss
        return weight_tensor - np.multiply(self.learning_rate, gradient_tensor)