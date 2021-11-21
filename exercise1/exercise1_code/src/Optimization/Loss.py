import numpy as np

class CrossEntropyLoss:

    EPSILON = np.finfo(float).eps

    def __init__(self):
        self.y_cap = None

    def forward(self, prediction_tensor, label_tensor):
        # result of last layer
        self.y_cap = prediction_tensor
        # actual result
        self.y = label_tensor
        return -np.sum(self.y * np.log(self.y_cap + CrossEntropyLoss.EPSILON))

    def backward(self, label_tensor):
        cel_der = - label_tensor / self.y_cap
        return cel_der