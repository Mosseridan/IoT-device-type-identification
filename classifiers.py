import pandas as pd
import numpy
from os import path
from sklearn import tree, ensemble
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from device_session_regressor import DeviceSessionRegressor
from device_session_classifier import DeviceSessionClassifier
from device_sequence_classifier import DeviceSequenceClassifier


def is_dev(this_dev_name, dev_name):
    return 1 if this_dev_name == dev_name else 0


def get_is_dev_vec(this_dev_name, dev_names):
    return [is_dev(this_dev_name, dev_name) for dev_name in dev_names]

def clear_missing_data(x_train, y_train):
    """ 
    This method is used to remove all the instances (examples)
    in which there is data missing (marked by question marks). In case of 
    even one missing value in an example, we remove the entire example from
    the dataset 
    """
    x_train = x_train.replace("?", numpy.NaN)
    dropped_rows = x_train.index[x_train.isnull().any(1)].tolist()
    x_train = x_train.dropna(0)
    new_y_train = []
    for i in range(0,len(y_train)):
        if i in dropped_rows:
            continue
        else:
            new_y_train.append(y_train[i])
    return x_train, new_y_train

def perform_feature_scaling(x_train):
    """
    This method is used in order to perform feature scaling according to the 
    min-max scaler. The scaler can be replaced with another one, like the
    standard scaler 
    """
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train)
    


y_col = 'device_category'
cols_to_drop = ['device_category']
use_cols = ['ttl_A_avg', 'ttl_A_entropy', 'ttl_A_firstQ', 'ttl_A_max', 'ttl_A_median', 'ttl_A_min', 'ttl_A_stdev',
            'ttl_A_sum', 'ttl_A_thirdQ', 'ttl_A_var', 'ttl_B_avg', 'ttl_B_entropy', 'ttl_B_firstQ', 'ttl_B_max',
            'ttl_B_median', 'ttl_B_min', 'ttl_B_stdev', 'ttl_B_sum', 'ttl_B_thirdQ', 'ttl_B_var', 'ttl_avg',
            'ttl_entropy', 'ttl_firstQ', 'ttl_max', 'ttl_median', 'ttl_min', 'ttl_stdev', 'ttl_sum', 'ttl_thirdQ',
            'ttl_var', 'is_ssl', 'is_http', 'is_g_http', 'is_cdn_http', 'is_img_http', 'is_ad_http',
            'is_numeric_url_http', 'is_numeric_url_with_port_http','is_tv_http', 'is_cloud_http', 'B_is_system_port',
            'B_is_user_port', 'B_is_dynamic_and_or_private_port', 'B_port_is_11095', 'B_port_is_1900', 'B_port_is_5222',
            'B_port_is_5223', 'B_port_is_5228', 'B_port_is_54975', 'B_port_is_80', 'B_port_is_8080', 'B_port_is_8280',
            'B_port_is_9543', 'B_port_is_else', 'device_category']

train = pd.read_csv(path.abspath('data/train.csv'), usecols=use_cols, low_memory=False)
#validation = pd.read_csv(path.abspath('data/validation.csv'), usecols=use_cols, low_memory=False,nrows=10000)
# test = pd.read_csv(path.abspath('data/test.csv'), low_memory=False)

x_train = train.drop(cols_to_drop, 1)
y_train = pd.Series(get_is_dev_vec('security_camera', train[y_col]))
x_train, y_train = clear_missing_data(x_train, y_train)
x_train = perform_feature_scaling(x_train)
        

# x_validation = validation.drop(cols_to_drop, 1)
# y_validation = validation[y_col]
# x_test = test.drop(cols_to_drop, 1)
# y_test = test[y_col]

cart = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5)
# rus = make_pipeline(RandomUnderSampler(),tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5))
# forest = ensemble.RandomForestClassifier(criterion='entropy', max_depth=15, min_samples_leaf=5)
# gboost = ensemble.GradientBoostingClassifier(max_depth=15, min_samples_leaf=5)
#
#cart.fit(x_train, y_train)
# rus.fit(x_train, y_train)
# forest.fit(x_train, y_train)
#gboost.fit(X_train, y_train)

# Usage example of the device session regressor 
# device_session_regressor = DeviceSessionRegressor('security_camera')
# cart = device_session_regressor.train(cart, x_train, y_train)
# print(device_session_regressor.predict(cart, [x_train[10]]))

# Usage example of the device session classifier 
#device_session_classifier = DeviceSessionClassifier('security_camera')
#cart = device_session_classifier.train(cart, x_train, y_train)
#print(device_session_classifier.predict(cart, [x_train[10]]))

# Usage example of the device sequence classifier 
#device_sequence_classifier = DeviceSequenceClassifier('security_camera')
#cart = device_sequence_classifier.train(cart, x_train, y_train, train)
#print(device_sequence_classifier.predict(cart, [x_train]))


# n_splits = 10
#
# ub = BaggingClassifier(warm_start=True, n_estimators=0)
#
# for split in range(n_splits):
#     X_res, y_res = RandomUnderSampler(random_state=split).fit_sample(X_train,y_train)
#     ub.n_estimators += 1
#     ub.fit(X_res, y_res)
#
# def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
#     from sklearn.metrics import roc_curve, roc_auc_score
#     fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
#     ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
#             label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))
#
# f, ax = plt.subplots(figsize=(6,6))
#
# roc_auc_plot(y_test,ub.predict_proba(X_test),label='UB ',l='-')
# roc_auc_plot(y_test,forest.predict_proba(X_test),label='FOREST ',l='--')
# roc_auc_plot(y_test,cart.predict_proba(X_test),label='CART', l='-.')
# roc_auc_plot(y_test,rus.predict_proba(X_test),label='RUS',l=':')
#
# ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--',
#         label='Random Classifier')
# ax.legend(loc="lower right")
# ax.set_xlabel('False Positive Rate')
# ax.set_ylabel('True Positive Rate')
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.set_title('Receiver Operator Characteristic curves')
# sns.despine()

print("@@@ DONE @@@")
