from device_classifier import DeviceClassifier


class MultipleDeviceClassifier:
    """
    A multi-class classifier used for determining the type of a device according to given sessions
    """

    def __init__(self,
                 dev_model_dict,
                 is_model_pkl=False,
                 use_cols=None,
                 y_col=None,
                 train=None,
                 is_train_csv=False,
                 validation=None,
                 is_validation_csv=False):

        self.dev_classifiers = [
            DeviceClassifier(
                dev_name,
                dev_model,
                is_model_pkl=is_model_pkl,
                use_cols=use_cols,
                y_col=y_col,
                train=train,
                is_train_csv=is_train_csv,
                validation=validation,
                is_validation_csv=is_validation_csv)
            for dev_name, dev_model in dev_model_dict.items()]

    def train(self, train, validation):
        for dev_classifier in self.dev_classifiers:
            dev_classifier.train(train, validation)

    def predict(self, devs_sessions):
        """ This method returns the name of the device the sessions originated from or None in case of an unknown device"""
        return [self.predict_dev(dev_sessions) for dev_sessions in devs_sessions]

    def predict_dev(self, dev_sessions):
        """ This method returns the name of the device the sessions originated from or None in case of an unknown device"""
        for dev_classifier in self.dev_classifiers:
            if dev_classifier.predict([dev_sessions])[0]:
                return dev_classifier.dev_name
        return None
