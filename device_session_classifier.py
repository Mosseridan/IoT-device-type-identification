import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


class DeviceSessionClassifier:
    """ Classifier used for determining whether a given session originated from a specific device or not """

    def __init__(self,
                 dev_name,
                 model,
                 is_model_pkl=False,
                 use_cols=None,
                 cols_to_drop=None,
                 y_col=None,
                 train=None,
                 is_train_csv=False):
        self.dev_name = dev_name
        if is_model_pkl:
            self.model = self.load_model_from_pkl(model)
        else:
            self.model = model
        if use_cols:
            self.use_cols = use_cols
        else:
            self.use_cols = pd.read_csv(os.path.abspath('data/use_cols.csv'))
        if cols_to_drop:
            self.cols_to_drop = cols_to_drop
        else:
            self.cols_to_drop = ['device_category']
        if y_col:
            self.y_col = y_col
        else:
            self.y_col = 'device_category'
        if train:
            if is_train_csv:
                train = self.load_data_from_csv(train)
            self.train(train)

    def is_dev(self, dev_name):
        return 1 if dev_name == self.dev_name else 0

    def get_is_dev_vec(self, dev_names):
        return [self.is_dev(dev_name) for dev_name in dev_names]

    def train(self, train):
        x_train, y_train = self.split_data(train)
        self.model.fit(x_train, y_train)

    def predict(self, sessions):
        if isinstance(sessions, pd.DataFrame):
            return self.model.predict(sessions.values)
        else:
            return self.model.predict(sessions)

    def split_data(self, data):
        x = self.perform_feature_scaling(self.test.drop(self.cols_to_drop, axis=1))
        y = self.test[self.y_col]
        return x, y

    def perform_feature_scaling(self, x_train):
        """
        This method is used in order to perform feature scaling according to the
        min-max scaler. The scaler can be replaced with another one, like the
        standard scaler
        """
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        return pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)

    def load_model_from_pkl(self, pkl):
        return joblib.load(os.path.abspath(pkl))

    def load_data_from_csv(self, csv):
        return pd.read_csv(os.path.abspath(csv), usecols=self.use_cols, low_memory=False)
