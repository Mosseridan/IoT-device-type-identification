import numpy as np
import pandas as pd
import utils
from device_session_classifier import DeviceSessionClassifier

class DeviceSequenceClassifier(DeviceSessionClassifier):
    """ A classifier used for determining whether a given sequence of sessions was originated from a specifc device or not """

    def __init__(self,
                 dev_name,
                 model,
                 is_model_pkl=False,
                 opt_seq_len=1,
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
                         is_train_csv=is_train_csv)
        if train and validation:
            if is_validation_csv:
                validation = utils.load_data_from_csv(validation, use_cols)
            self.find_opt_seq_len(validation)
        else:
            self.opt_seq_len = opt_seq_len

    def train(self, train, validation):
        super().train(train)
        self.opt_seq_len = self.find_opt_seq_len(validation)

    def predict(self, sequences):
        if isinstance(sequences, pd.DataFrame):
            return sequences.apply(self.predict_sequence, axis=1).values
        else:
            return np.array([self.predict_sequence(seq) for seq in sequences])

    def predict_sequence(self, sequence):
        sess_preds = super().predict(sequence)
        seq_pred = 1 if sess_preds.sum() > (len(sess_preds) / 2) else 0
        return seq_pred

    def find_opt_seq_len(self, validation, update=True):
        # Finds minimal seq length s.t accuracy=1 on all sessions
        opt_seq_len = 1
        # Find minimal sequence length s.t FPR=1 for all other devs
        for dev_name, dev_sessions in validation.groupby('device_category'):
            dev_sessions = utils.split_data(dev_sessions, y_col=self.y_col)[0]
            is_dev = self.is_dev(dev_name)
            start = 0
            seq_len = 1
            while start + seq_len < len(dev_sessions):
                is_dev_pred = self.predict([dev_sessions[start:start + seq_len]])
                if is_dev == is_dev_pred:
                    start += 1
                else:
                    start = max(1, start-2)
                    seq_len += 2
            opt_seq_len = max(seq_len, opt_seq_len)
        # Return minimal seq length s.t accuracy=1
        if update:
            self.opt_seq_len = opt_seq_len
        return opt_seq_len

    def get_sub_sequences(self, sessions, seq_len=None):
        if not seq_len:
            seq_len = self.opt_seq_len
        return utils.get_sub_sequences(sessions, seq_len)

    def split_data(self, data):
        data = utils.clear_missing_data(data)
        x = []
        y = []
        for dev_name, dev_sessions in data.groupby(self.y_col):
            dev_sessions = utils.perform_feature_scaling(dev_sessions)
            is_dev = self.is_dev(dev_name)
            seqs = self.get_sub_sequences(dev_sessions)
            x += seqs
            y += [is_dev]*len(seqs)
        return x, y
