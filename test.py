import os
import  pandas as pd
from experiments_manager import ExperimentsManager
from device_session_classifier import DeviceSessionClassifier
from device_sequence_classifier import DeviceSequenceClassifier

device = 'watch'
other_device = 'motion_sensor'
model_pkl = r'models/{0}/{0}_cart_entropy_50_samples_leaf.pkl'.format(device)
dataset_csv = 'data/validation.csv'
sc = DeviceSequenceClassifier(device, model_pkl, is_model_pkl=True)


m = sc.load_model_from_pkl(model_pkl)

validation = sc.load_data_from_csv(dataset_csv)


all_sess = validation.drop(sc.y_col, axis=1).values
other_dev_sess = validation.groupby(sc.y_col).get_group(other_device)
other_dev_sess = other_dev_sess.drop(sc.y_col, axis=1).values

sc.find_opt_seq_len(validation)
seq_len = 7
seqs = [other_dev_sess[i:i+seq_len] for i in range(len(other_dev_sess)-seq_len)]

# seq_len = 1
# seqs = [sessions[0:seq_len]]

x = m.predict(other_dev_sess)
pred = sc.predict(seqs)
print(pred)
print("YAYY!!")