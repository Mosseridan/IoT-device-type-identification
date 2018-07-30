from device_classifier import DeviceClassifier


class MultipleDeviceClassifier():

    def __init__(self, train=None, validation=None):
        self.dev_names = train['device_category'].unique()
        self.dev_classifiers = [DeviceClassifier(dev_name, train, validation) for dev_name in self.dev_names]

    def train(self, train, validation):
        for dev_classifier in self.dev_classifiers:
            dev_classifier.train(train, validation)

    def predict(self, sessions):
        # Returns name of the device the session originated from or None for an unknown device
        for dev_classifier in self.dev_classifiers:
            if dev_classifier.predict(sessions):
                return dev_classifier.dev
        return None
