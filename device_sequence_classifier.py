from device_session_classifier import DeviceSessionClassifier


class DeviceSequenceClassifier(DeviceSessionClassifier):
    """ A classifier used for determining whether a given sequence of sessions was originated from a specifc device or not """

    def __init__(self, dev_name):
        super().__init__(dev_name)
        self.opt_seq_len = 1

    def train(self, model, x_train, y_train, validation):
        model = super().train(model, x_train, y_train)
        #self.opt_seq_len = self.find_opt_seq_len(model, validation)
        return model 

    def predict(self, model, sequence):
        predictions_sum = 0
        for session in sequence:
            predictions_sum += super().predict(model, session)
        return 1 if predictions_sum > (len(sequence) / 2) else 0 

    def find_opt_seq_len(self, model, validation):
        # Finds minimal seq length s.t accuracy=1 on all sessions
        opt_seq_len = 1
        # Find minimal sequence length s.t FPR=1 for all other devs
        for dev_name, dev_sessions in validation.groupby('device_category'):
            start = 1
            seq_len = 1
            while start + seq_len <= len(dev_sessions):
                is_dev = dev_name == self.dev_name
                print("dev_sessions {}" .format(dev_sessions))
                is_dev_pred = self.predict(model, dev_sessions[start:start + seq_len])
                if is_dev == is_dev_pred:
                    start += 1
                else:
                    start = 1
                    seq_len += 2
            opt_seq_len = max(seq_len, opt_seq_len)
        # Return minimal seq length s.t accuracy=1
        return opt_seq_len
