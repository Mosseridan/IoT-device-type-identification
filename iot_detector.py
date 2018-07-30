import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer


def load_data(data_path, label_col, feature_cols=None):
    # Load data csv with pandas and divide it to features and labels
    if feature_cols:
        data = pd.read_csv(os.path.abspath(data_path), usecols=feature_cols + [label_col], low_memory=False)
        features = data[feature_cols]
    else:
        data = pd.read_csv(os.path.abspath(data_path), usecols=None, low_memory=False)
        features = data.drop(label_col, 1)
    labels = data[label_col]
    return features, labels


def create_dev_sess_regressor(sessions, is_dev_labels):
    # Creates a regressor that returns the probability of a session originating from dev
    # TODO: Implement this! return regressor for device
    def dev_sess_regressor(sess):
        return sess / 10

    return dev_sess_regressor


def classify_sess(dev_sess_regressor, threshold, sess):
    # Classifies a session with regressor according to threshold
    return 1 if dev_sess_regressor(sess) > threshold else 0


def create_dev_sess_classifier(dev_sess_regressor, threshold):
    # Creates a classifier that returns 1 if a session originates from dev and 0 otherwise
    def dev_sess_classifier(sess):
        return classify_sess(dev_sess_regressor, threshold, sess)

    return dev_sess_classifier


def find_opt_threshold(dev_sess_regressor, sessions, is_dev_labels):
    # TODO: Implement this! returns optimal threshold for device classefication with given regressor
    return 0.5


def classify_seq(dev_sess_classifier, seq):
    # Classifies a seq with classifier according to a majority vote
    return 1 if sum(map(dev_sess_classifier, seq)) > len(seq) / 2 else 0


def create_dev_seq_classifier(dev_sess_classifier):
    # Creates a classifier that returns 1 if a majority of sessions in a sequence originate from dev and 0 otherwise
    def dev_seq_classifier(seq):
        return classify_seq(dev_sess_classifier, seq)

    return dev_seq_classifier


def classify_dev(dev_sess_classifier, opt_seq_len, sessions):
    for start in range(len(sessions) - opt_seq_len):
        if classify_seq(dev_sess_classifier, sessions[start:start + opt_seq_len]):
            return 1
    return 0


def create_dev_classifier(dev_sess_classifier, opt_seq_len):
    def dev_classifier(sessions):
        return classify_dev(dev_sess_classifier, opt_seq_len, sessions)

    return dev_classifier


def find_opt_seq_len(this_dev, dev_sess_classifier, dev_sess_dict):
    # Finds minimal seq length s.t accuracy=1 on all sessions
    opt_seq_len = 1
    # Find minimal sequence length s.t FPR=1 for all other devs
    for dev, dev_sess in dev_sess_dict.items():
        start = 1
        seq_len = 1
        while start + seq_len <= len(dev_sess):
            is_dev = dev == this_dev
            is_dev_pred = classify_seq(dev_sess_classifier, dev_sess[start:start + seq_len])
            if is_dev == is_dev_pred:
                start += 1
            else:
                start = 1
                seq_len += 2
        opt_seq_len = max(seq_len, opt_seq_len)
    # Return minimal seq length s.t accuracy=1
    return opt_seq_len


# def find_opt_seq_len(dev_sess_classifier, dev_sessions, other_devs_sessions):
#     # Finds minimal seq length s.t accuracy=1 on all sessions
#     opt_seq_len = 1
#     # Find minimal sequence length s.t TPR=1
#     start = 0
#     seq_len = 1
#     while start + seq_len <= len(dev_sessions):
#         if classify_seq(dev_sess_classifier, dev_sessions[start:start + seq_len]):
#             start += 1
#         else:
#             start = 1
#             seq_len += 2
#     opt_seq_len = max(seq_len, opt_seq_len)
#     # Find minimal sequence length s.t FPR=1 for all other devs
#     for other_dev_sessions in other_devs_sessions:
#         start = 1
#         seq_len = 1
#         while start+seq_len <= len(other_dev_sessions):
#             if classify_seq(dev_sess_classifier, other_dev_sessions[start:start + seq_len]):
#                 start = 1
#                 seq_len += 2
#             else:
#                 start += 1
#         opt_seq_len = max(seq_len, opt_seq_len)
#     # Return minimal seq length s.t accuracy=1
#     return opt_seq_len


def classify_multi_dev(dev_cls_dict, dev_sessions):
    # Returns name of the device the session originated from or None for an unknown device
    for dev, dev_classifier in dev_cls_dict.items():
        if dev_classifier(dev_sessions):
            return dev
    return None


def create_multi_dev_classifier(dev_cls_dict):
    def multi_dev_classifier(dev_sessions):
        return classify_multi_dev(dev_cls_dict, dev_sessions)

    return multi_dev_classifier


def is_eq(a):
    return lambda b: 1 if a == b else 0


def create_iot_classifier(train, validation):
    train_sessions = train.drop('device_category')
    validation_sessions = validation.drop('device_category')
    devs = train['device_category'].unique()
    train_is_devt_dict = {dev: train['device_category'].apply(is_eq(dev)) for dev in devs}
    validation_is_devt_dict = {dev: validation['device_category'].apply(is_eq(dev)) for dev in devs}
    dev_sess_dict = {dev: sess for dev, sess in train.groupby('device_category')}
    dev_sess_reg_dict = {dev: create_dev_sess_regressor(train_sessions, train_is_devt_dict[dev]) for dev in devs}
    opt_thr_dict = {dev: find_opt_threshold(dev_sess_reg_dict[dev], validation_sessions, validation_is_devt_dict[dev])
                    for dev, is_dev in devs}
    dev_sess_cls_dict = {dev: create_dev_sess_classifier(dev_sess_reg_dict[dev], opt_thr_dict[dev]) for dev in devs}
    opt_seq_len_dict = {dev: find_opt_seq_len(dev, dev_sess_cls_dict[dev], dev_sess_dict) for dev in devs}
    dev_cls_dict = {dev: create_dev_classifier(dev_sess_cls_dict[dev], opt_seq_len_dict[dev]) for dev in devs}

    return create_multi_dev_classifier(dev_cls_dict)


train = pd.read_csv(os.path.abspath('data/train.csv'), usecols=['ack', 'device_category'], low_memory=False)
validation = pd.read_csv(os.path.abspath('data/validation.csv'), usecols=['ack', 'device_category'], low_memory=False)
test = pd.read_csv(os.path.abspath('data/test.csv'), usecols=['ack', 'device_category'], low_memory=False)

classifier = create_iot_classifier(train, validation)

print('@@ DONE @@')
