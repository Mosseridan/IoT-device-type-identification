from device_sequence_classifier import DeviceSequenceClassifier


class DeviceClassifier(DeviceSequenceClassifier):

    def __init__(self, dev_name, train=None, validation=None):
        DeviceSequenceClassifier(dev_name, train, validation)

    def train(self, train, validation):
        super().train(train, validation)

    def predict(self, sessions):
        for start in range(len(sessions) - super().opt_seq_len):
            if super().predict(sessions[start:start + super().opt_seq_len]):
                return 1
        return 0
