import os
import utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from experiments_manager import ExperimentsManager
from sklearn.preprocessing import MinMaxScaler
from device_session_classifier import DeviceSessionClassifier
from device_sequence_classifier import DeviceSequenceClassifier
from sklearn import metrics

sns.set_style("white")

def eval_classifier(
        classifier,
        dataset,
        model_name,
        dataset_name,
        classification_method,
        seq_len,
        opt_seq_len,
        metrics_headers,
        metrics_csv):
    dataset_metrics = classifier.eval_on_dataset(dataset)
    dataset_metrics['model'] = model_name
    dataset_metrics['dataset'] = dataset_name
    dataset_metrics['classification_method'] = classification_method
    dataset_metrics['seq_len'] = seq_len
    dataset_metrics['opt_seq_len'] = opt_seq_len
    if os.path.exists(metrics_csv):
        header = False
    else:
        header = metrics_headers
    # dataset_metrics_df = pd.DataFrame({k: [v] for k, v in dataset_metrics.items()}, columns=metrics_headers)
    dataset_metrics_df = pd.DataFrame(dataset_metrics, columns=metrics_headers)
    dataset_metrics_df.to_csv(metrics_csv, mode='a', header=header, index=False)


y_col = 'device_category'
use_cols = pd.read_csv(os.path.abspath('data/use_cols.csv'))
datasets = ['test','validation','train']
devices = list(pd.read_csv(os.path.abspath('data/devices.csv')))
metrics_headers = list(pd.read_csv(os.path.abspath('data/metrics_headers.csv')))
models_dir = os.path.abspath('models')
metrics_dir = os.path.abspath('metrics')
os.makedirs(metrics_dir, exist_ok=True)
metrics_csv = os.path.join(metrics_dir, 'metrics.csv')

for dataset_name in datasets:
    print('@', dataset_name)
    dataset = utils.load_data_from_csv('data/{}.csv'.format(dataset_name), use_cols=use_cols)
    for dev_name in devices:
        if dev_name == 'thermostat':
            continue

        for model_pkl in os.listdir(os.path.join(models_dir,dev_name)):
            model_name = os.path.splitext(model_pkl)[0]
            print('@@', dev_name, model_name)

            # Device session classifier
            print("@@@ Device session classifier")
            dev_sess_cls = DeviceSessionClassifier(
                dev_name,
                os.path.join(models_dir, dev_name, model_pkl),
                is_model_pkl=True,
                use_cols=use_cols,
                y_col=y_col)

            eval_classifier(
                classifier=dev_sess_cls,
                dataset=dataset,
                model_name=model_name,
                dataset_name=dataset_name,
                classification_method='session',
                seq_len=1,
                opt_seq_len=1,
                metrics_headers=metrics_headers,
                metrics_csv=metrics_csv)

            # Device sequence classifier
            print("@@@ Device sequence classifier")
            dev_seq_cls = DeviceSequenceClassifier(
                dev_name,
                os.path.join(models_dir, dev_name, model_pkl),
                is_model_pkl=True,
                use_cols=use_cols,
                y_col=y_col)

            if dataset_name == 'train':
                update = True
            else:
                update = False
            opt_seq_len = dev_seq_cls.find_opt_seq_len(dataset, update=update)
            print('{} seq_length: {}, optimal sequence length: {}'
                  .format(dataset_name, dev_seq_cls.opt_seq_len, opt_seq_len))
            eval_classifier(
                classifier=dev_sess_cls,
                dataset=dataset,
                model_name=model_name,
                dataset_name=dataset_name,
                classification_method='sequence',
                seq_len=dev_seq_cls.opt_seq_len,
                opt_seq_len=opt_seq_len,
                metrics_headers=metrics_headers,
                metrics_csv=metrics_csv)


print('YAYYY!!!')
#
# device = 'watch'
# other_device = 'watch'
# model_pkl = r'models/{0}/{0}_cart_entropy_100_samples_leaf.pkl'.format(device)
# dataset_csv = 'data/validation.csv'
# sc = DeviceSequenceClassifier(device, model_pkl, is_model_pkl=True)
#
# def perform_feature_scaling(x_train):
#     scaler = MinMaxScaler()
#     scaler.fit(x_train)
#     return scaler.transform(x_train)
#
#
# m = sc.load_model_from_pkl(model_pkl)
#
# validation = sc.load_data_from_csv(dataset_csv)
#
#
# all_sess = sc.split_data(validation)[0]
# other_dev_sess = validation.groupby(sc.y_col).get_group(other_device)
# other_dev_sess = sc.split_data(other_dev_sess)[0]
#
# opt_seq_len = sc.find_opt_seq_len(validation)
# seqs = [other_dev_sess[i:i+seq_len] for i in range(len(other_dev_sess)-seq_len)]
#
# # seq_len = 1
# # seqs = [sessions[0:seq_len]]
#
# classification = 1 if device == other_device else 0
#
# y_actual = [classification] * 100
# x = m.predict(other_dev_sess)
# print(metrics.accuracy_score(y_actual,x))
# pred = sc.predict(seqs)
# y_actual = [classification] * len(seqs)
# print(metrics.accuracy_score(y_actual,pred))
# print(pred)
# print("YAYY!!")