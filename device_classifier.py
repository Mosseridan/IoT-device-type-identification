import numpy as np
import utils
from device_sequence_classifier import DeviceSequenceClassifier


class DeviceClassifier(DeviceSequenceClassifier):
    """ A classifier used for determining whether some given sessions were originated from the given device or not """

    def __init__(self,
                 dev_name,
                 model,
                 is_model_pkl=False,
                 use_cols=None,
                 y_col=None,
                 train=None,
                 is_train_csv=False,
                 validation=None,
                 is_validation_csv=False):
        super().__init__(dev_name=dev_name,
                         model=model,
                         is_model_pkl=is_model_pkl,
                         use_cols=use_cols,
                         y_col=y_col, train=train,
                         is_train_csv=is_train_csv,
                         validation=validation,
                         is_validation_csv=is_validation_csv)

    def train(self, train, validation):
        super().train(train, validation)

    def predict(self, devs_sessions):
        for start in range(len(devs_sessions) - super().opt_seq_len):
            if super().predict([devs_sessions[start:start + super().opt_seq_len]]):
                return 1
        return 0

    # def predict(self, devs_sessions):
    #     return super.predict([self.choose_rand_seq(dev_sessions) for dev_sessions in devs_sessions])

    def choose_rand_seq(self, dev_sessions):
        start = np.random.randint(len(dev_sessions) - super().opt_seq_len)
        return dev_sessions[start:start+super().opt_seq_len]