import numpy as np
import pandas as pd

class DeviceSessionRegressor(object):
    def __init__(self, dev_name, train=None, validation=None):
        self.dev_name = dev_name
        if train and validation:
            self.train(train, validation)

    def is_dev(self, dev_name):
        return 1 if dev_name == self.dev_name else 0

    def get_is_dev_vec(self, dev_names):
        return [self.is_dev(dev_name) for dev_name in dev_names]

    def train(self, train, validation):
    # TODO: Implement this: trains device regression model
        return

    def predict(self, train, validation):
    # TODO: Implement this: predicts probability of session originating from this dev
        return np.random.random()

