import numpy as np

# Base class for training layers
class BaseLayer:
    def __init__(self):
        self.trainable = False