from device_session_regressor import DeviceSessionRegressor


class DeviceSessionClassifier(DeviceSessionRegressor):
    """ Classifier used for determining whether a given session originated from a specifc device or not """
    
    def __init__(self, dev_name):
        super().__init__(dev_name)
        self.threshold = 0.5

    def train(self, model, x_train, y_train):
        self.threshold = self.find_opt_threshold()
        return super().train(model, x_train, y_train)

    def predict(self, model, session):
        return 1 if super().predict(model, session) > self.threshold else 0

    def find_opt_threshold(self):
        # TODO: Implement this! returns optimal threshold for device classefication with given regressor
        return 0.5
