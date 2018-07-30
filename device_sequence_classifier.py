from device_session_classidier import DeviceSessionClassifier


class DeviceSequenceClassifier(DeviceSessionClassifier):

    def __init__(self, dev_name, train=None, validation=None):
        DeviceSessionClassifier(dev_name)
        self.opt_seq_len = 1
        if train and validation:
            self.train(train, validation)

    def train(self, train, validation):
        super().train(train, validation)
        self.opt_seq_len = self.find_opt_seq_len(validation)

    def predict(self, sequence):
        return 1 if sum(map(super().predict, sequence)) > len(sequence) / 2 else 0

    def find_opt_seq_len(self, validation):
        # Finds minimal seq length s.t accuracy=1 on all sessions
        opt_seq_len = 1
        # Find minimal sequence length s.t FPR=1 for all other devs
        for dev_name, dev_sessions in validation.groupby('device_category'):
            start = 1
            seq_len = 1
            while start + seq_len <= len(dev_sessions):
                is_dev = dev_name == self.dev_name
                is_dev_pred = self.predict(dev_sessions[start:start + seq_len])
                if is_dev == is_dev_pred:
                    start += 1
                else:
                    start = 1
                    seq_len += 2
            opt_seq_len = max(seq_len, opt_seq_len)
        # Return minimal seq length s.t accuracy=1
        return opt_seq_len