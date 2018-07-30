from device_session_regressor import DeviceSessionRegressor


class DeviceSessionClassifier(DeviceSessionRegressor):
    def __init__(self, dev_name, train=None, validation=None):
        DeviceSessionRegressor(dev_name)
        self.threshold = 0.5
        if train and validation:
            self.train(train, validation)

    def train(self, train, validation):
        super().train(train, validation)
        self.threshold = self.find_opt_threshold(validation)

    def predict(self, session):
        return 1 if super().predict(session) > self.threshold else 0

    def find_opt_threshold(self, validation):
        # TODO: Implement this! returns optimal threshold for device classefication with given regressor
        return 0.5