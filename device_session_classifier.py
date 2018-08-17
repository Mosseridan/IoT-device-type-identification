import os
import numpy as np
import pandas as pd
import utils

class DeviceSessionClassifier:
    """ Classifier used for determining whether a given session originated from a specific device or not """

    def __init__(self,
                 dev_name,
                 model,
                 is_model_pkl=False,
                 use_cols=None,
                 y_col=None,
                 train=None,
                 is_train_csv=False):
        self.dev_name = dev_name
        if is_model_pkl:
            self.load_model_from_pkl(model)
        else:
            self.model = model
        if use_cols is not None:
            self.use_cols = use_cols
        else:
            self.use_cols = pd.read_csv(os.path.abspath('data/use_cols.csv'))
        if y_col:
            self.y_col = y_col
        else:
            self.y_col = 'device_category'
        if train:
            if is_train_csv:
                train = utils.load_data_from_csv(train, use_cols)
            elif use_cols:
                train = train[use_cols]
            self.train(train)

    def is_dev(self, dev_name):
        return 1 if dev_name == self.dev_name else 0

    def get_is_dev_vec(self, dev_names):
        return [self.is_dev(dev_name) for dev_name in dev_names]

    def train(self, train):
        x_train, y_train = utils.split_data(train, y_col=self.y_col)
        self.model.fit(x_train, y_train)

    def predict(self, sessions):
        if isinstance(sessions, pd.DataFrame):
            return self.model.predict(sessions.values)
        else:
            return self.model.predict(sessions)

    def load_model_from_pkl(self, pkl):
        self.model = utils.load_model_from_pkl(pkl)

    def split_data(self, data):
        x, y = utils.split_data(data, self.y_col)
        y = self.get_is_dev_vec(y)
        return x, y

    def eval_on_dataset(self, dataset, is_dataset_csv=False):
        if is_dataset_csv:
            dataset = utils.load_data_from_csv(dataset, self.use_cols)
        # Split data to features and labels
        x, y_true = self.split_data(dataset)
        # Classify data
        y_pred = self.predict(x)
        # Evaluate predictions
        return utils.eval_predictions(y_true, y_pred)
