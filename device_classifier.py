import pandas as pd
from device_sequence_classifier import DeviceSequenceClassifier


class DeviceClassifier(DeviceSequenceClassifier):
    """ A classifier used for determining whether some given sessions were originated from the given device or not """

    def __init__(self,
                 dev_name,
                 model,
                 is_model_pkl=False,
                 use_cols=None,
                 cols_to_drop=None,
                 y_col=None,
                 train=None,
                 is_train_csv=False,
                 validation=None,
                 is_validation_csv=False):
        super().__init__(dev_name=dev_name,
                         model=model,
                         is_model_pkl=is_model_pkl,
                         use_cols=use_cols,
                         cols_to_drop=cols_to_drop,
                         y_col=y_col, train=train,
                         is_train_csv=is_train_csv,
                         validation=validation,
                         is_validation_csv=is_validation_csv)

    def train(self, train, validation):
        super().train(train, validation)

    def predict(self, sessions):
        for start in range(len(sessions) - super().opt_seq_len):
            if super().predict([sessions[start:start + super().opt_seq_len]]):
                return 1
        return 0
