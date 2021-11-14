import numpy as np

# Base class for training layers
class BaseLayer:
    def __init__(self):
        self.trainable = False
        #Optionally, you can add other members like a default weights parameter, which might
#come in handy
