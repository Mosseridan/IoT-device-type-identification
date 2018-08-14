import numpy as np
import pandas as pd
from device_session_classifier import DeviceSessionClassifier

class DeviceSequenceClassifier(DeviceSessionClassifier):
    """ A classifier used for determining whether a given sequence of sessions was originated from a specifc device or not """

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
                         is_train_csv=is_train_csv)
        if train and validation:
            if is_validation_csv:
                validation = self.load_data_from_csv(validation)
            self.opt_seq_len = self.find_opt_seq_len(validation)
        else:
            self.opt_seq_len = 1

    def train(self, train, validation):
        super().train(train)
        self.opt_seq_len = self.find_opt_seq_len(validation)

    def predict(self, sequences):
        if isinstance(sequences, pd.DataFrame):
            return sequences.apply(self.predict_sequence, axis=1).values
        else:
            return np.array([self.predict_sequence(seq) for seq in sequences])

    def predict_sequence(self, sequence):
        return 1 if super().predict(sequence).sum() > (len(sequence) / 2) else 0

    def find_opt_seq_len(self, validation):
        # Finds minimal seq length s.t accuracy=1 on all sessions
        opt_seq_len = 1
        # Find minimal sequence length s.t FPR=1 for all other devs
        for dev_name, dev_sessions in validation.groupby('device_category'):
            dev_sessions = dev_sessions.drop(self.y_col, axis=1).values
            is_dev = self.is_dev(dev_name)
            start = 0
            seq_len = 1
            while start + seq_len <= len(dev_sessions):
                is_dev_pred = self.predict([dev_sessions[start:start + seq_len]])
                if is_dev == is_dev_pred:
                    start += 1
                else:
                    start = max(1,start-1)
                    seq_len += 2
            opt_seq_len = max(seq_len, opt_seq_len)
        # Return minimal seq length s.t accuracy=1
        self.opt_seq_len = opt_seq_len
        return opt_seq_len
