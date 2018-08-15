import os
import  pandas as pd
from experiments_manager import ExperimentsManager
from sklearn.preprocessing import MinMaxScaler
from device_session_classifier import DeviceSessionClassifier
from device_sequence_classifier import DeviceSequenceClassifier
from sklearn import metrics

device = 'watch'
other_device = 'motion_sensor'
model_pkl = r'models/{0}/{0}_cart_entropy_50_samples_leaf.pkl'.format(device)
dataset_csv = 'data/validation.csv'
sc = DeviceSequenceClassifier(device, model_pkl, is_model_pkl=True)

def perform_feature_scaling(x_train):
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train)


m = sc.load_model_from_pkl(model_pkl)

validation = sc.load_data_from_csv(dataset_csv)


all_sess = validation.drop(sc.y_col, axis=1).values
all_sess = perform_feature_scaling(all_sess)
other_dev_sess = validation.groupby(sc.y_col).get_group(other_device)
other_dev_sess = other_dev_sess.drop(sc.y_col, axis=1).values
other_dev_sess = perform_feature_scaling(other_dev_sess)

sc.find_opt_seq_len(validation)
seq_len = 4
seqs = [other_dev_sess[i:i+seq_len] for i in range(len(other_dev_sess)-seq_len)]

# seq_len = 1
# seqs = [sessions[0:seq_len]]

classification = 1 if device == other_device else 0

y_actual = [classification] * 100
x = m.predict(other_dev_sess)
print(metrics.accuracy_score(y_actual,x))
pred = sc.predict(seqs)
y_actual = [classification] * len(seqs)
print(metrics.accuracy_score(y_actual,pred))
print(pred)
print("YAYY!!")