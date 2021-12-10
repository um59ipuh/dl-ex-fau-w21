import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.class_input_tensor = []
        self.class_output_tensor = []

    def forward(self, input_tensor, label_tensor):
        self.class_input_tensor = input_tensor

        array = input_tensor[label_tensor == 1]
        new_array = array + np.finfo(float).eps
        minus_log_array = np.log(new_array)
        loss = np.sum(-1 * minus_log_array)

        return loss

    def backward(self, label_tensor):
        error_tensor = np.true_divide(label_tensor, self.class_input_tensor) * -1
        return error_tensor