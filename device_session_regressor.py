import numpy as np
import pandas as pd

class DeviceSessionRegressor(object):
    """ A regressor used for predicting the probablity that a given session was originated from a specific device """
    
    def __init__(self, dev_name):
        self.dev_name = dev_name

    def is_dev(self, dev_name):
        return 1 if dev_name == self.dev_name else 0

    def get_is_dev_vec(self, dev_names):
        return [self.is_dev(dev_name) for dev_name in dev_names]
    
    def train(self, model, x_train, y_train):
        return model.fit(x_train, y_train)
  
    def predict(self, model, session):
        """ This method returns the probability that the given session (feature vector) was originated from the relevant
        device according to the given model """
        return model.predict_proba(session)[0][1]

