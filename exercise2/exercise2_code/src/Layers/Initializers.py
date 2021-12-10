import numpy as np


class Constant:
    def __init__(self, value = 0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value)


class UniformRandom:

    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        num = fan_in * fan_out
        uni_value = np.random.uniform(0, 1, num).reshape(weights_shape)
        return uni_value


class Xavier:

    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        theta = np.sqrt((2.0 / (fan_in + fan_out)))
        num = 1
        for size in weights_shape:
            num = num * size
        return np.random.normal(0, theta, num).reshape(weights_shape)


class He:

    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        theta = np.sqrt(2.0 / fan_in)
        num = 1
        for size in weights_shape:
            num = num * size
        return np.random.normal(0, theta, num).reshape(weights_shape)