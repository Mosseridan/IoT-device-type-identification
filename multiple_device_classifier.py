from device_classifier import DeviceClassifier


class MultipleDeviceClassifier:
    """
    A multi-class classifier used for determining the type of a device according to given sessions
    """

    def __init__(self, train=None, validation=None):
        self.dev_names = train['device_category'].unique()
        self.dev_classifiers = [DeviceClassifier(dev_name, train, validation) for dev_name in self.dev_names]

    def train(self, train, validation):
        for dev_classifier in self.dev_classifiers:
            dev_classifier.train(train, validation)

    def predict(self, sessions):
        """ This method returns the name of the device the sessions originated from or None in case of an unknown device"""
        for dev_classifier in self.dev_classifiers:
            if dev_classifier.predict(sessions):
                return dev_classifier.dev
        return None
