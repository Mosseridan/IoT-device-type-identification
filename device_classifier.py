import numpy as np
import utils
from device_sequence_classifier import DeviceSequenceClassifier


class DeviceClassifier(DeviceSequenceClassifier):
    """ A classifier used for determining whether some given sessions were originated from the given device or not """

    def __init__(self,
                 dev_name,
                 model,
                 is_model_pkl=False,
                 pred_method='all',
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

    def all_single_predict(self, dev_sessions):
        for start in range(len(dev_sessions) - super().opt_seq_len):
            if super().predict([dev_sessions[start:start + super().opt_seq_len]])[0]:
                return 1
        return 0

    def all_predict(self, devs_sessions):
        return [self.all_single_predict(dev_sessions) for dev_sessions in devs_sessions]

    def first_predict(self, devs_sessions):
        return super().predict([dev_sessions[0:self.opt_seq_len] for dev_sessions in devs_sessions])

    def random_predict(self, devs_sessions):
        return super().predict([self.choose_rand_seq(dev_sessions) for dev_sessions in devs_sessions])

    def predict(self, devs_sessions):
        pred_methods = {
            'all': self.all_predict,
            'first': self.first_predict,
            'random': self.random_predict
        }
        return pred_methods[self.pred_method](devs_sessions)


    # def predict(self, devs_sessions):
    #     return super.predict([self.choose_rand_seq(dev_sessions) for dev_sessions in devs_sessions])

    def choose_rand_seq(self, dev_sessions):
        start = np.random.randint(len(dev_sessions) - super().opt_seq_len)
        return dev_sessions[start:start+super().opt_seq_len]