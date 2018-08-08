import pandas as pd
import numpy
import os
from sklearn import tree, ensemble
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score

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

def clear_missing_data2(x_train, y_train):
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


def clear_missing_data(df):
    df_with_nan = df.replace("?", numpy.NaN)
    return df_with_nan.dropna(0)


def take_n_of_each(df, n):
    return df.groupby('device_category').head(n)


def perform_feature_scaling(x_train):
    """
    This method is used in order to perform feature scaling according to the 
    min-max scaler. The scaler can be replaced with another one, like the
    standard scaler 
    """
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train)
    

def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    score = roc_auc_score(y_true, y_proba[:, 1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)" % (label, score))
    return score


y_col = 'device_category'
cols_to_drop = ['device_category']
use_cols = pd.read_csv(os.path.abspath('data/train.csv'))


# Load Data
train = pd.read_csv(os.path.abspath('data/train.csv'), usecols=use_cols, low_memory=False)
# validation = pd.read_csv(os.path.abspath('data/validation.csv'), usecols=use_cols, low_memory=False)
test = pd.read_csv(os.path.abspath('data/test.csv'), usecols=use_cols, low_memory=False)

# Get device list
devices = train[y_col].unique()

# Shuffle Data
train = shuffle(train)
# validation = shuffle(validation)
test = shuffle(test)

# Remove missing data
train = clear_missing_data(train)
# validation = clear_missing_data(validation)
test = clear_missing_data(test)

# Seperate data to features
x_train = train.drop(cols_to_drop, 1)
# x_validation = validation.drop(cols_to_drop, 1)
x_test = test.drop(cols_to_drop, 1)

# Clean the data
x_train = perform_feature_scaling(x_train)
# x_validation = perform_feature_scaling(x_validation)
x_test = perform_feature_scaling(x_test)

roc_auc_scores = {}
for dev in devices:
    print("Learning models for device: {}".format(dev))

    # Get data labels
    y_train = pd.Series(get_is_dev_vec(dev, train[y_col]))
    # y_validation = pd.Series(get_is_dev_vec(dev, validation[y_col]))
    y_test = pd.Series(get_is_dev_vec(dev, test[y_col]))

    if y_train.sum() == 0:
        print('No {} sessions in training data. skipping model training.'.format(dev))
        continue

    # Create Models
    cart = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5)
    rus = make_pipeline(RandomUnderSampler(),tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5))
    forest = ensemble.RandomForestClassifier(criterion='entropy', max_depth=15, min_samples_leaf=5)
    gboost = ensemble.GradientBoostingClassifier(max_depth=15, min_samples_leaf=5)

    # Train Models
    cart.fit(x_train, y_train)
    rus.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    # gboost.fit(X_train, y_train)

    # Save Models
    dev_models_dir = os.path.abspath(os.path.join('models',dev))
    os.makedirs(dev_models_dir, exist_ok=True)
    joblib.dump(cart, os.path.join(dev_models_dir,'{}_cart.pkl'.format(dev)))

    if y_test.sum() == 0:
        print('No {} sessions in test data. skipping model evaluation.'.format(dev))
        continue


    # Plot ROC AUC curves
    n_splits = 10
    ub = BaggingClassifier(warm_start=True, n_estimators=0)

    for split in range(n_splits):
        x_res, y_res = RandomUnderSampler(random_state=split).fit_sample(x_train, y_train)
        ub.n_estimators += 1
        ub.fit(x_res, y_res)

    f, ax = plt.subplots(figsize=(6, 6))

    scores = []
    scores.append(roc_auc_plot(y_test, ub.predict_proba(x_test), label='UB ', l='-'))
    scores.append(roc_auc_plot(y_test, forest.predict_proba(x_test), label='FOREST ', l='--'))
    scores.append(roc_auc_plot(y_test, cart.predict_proba(x_test), label='CART', l='-.'))
    scores.append(roc_auc_plot(y_test, rus.predict_proba(x_test), label='RUS', l=':'))

    roc_auc_scores[dev] = scores

    ax.plot([0, 1], [0, 1], color='k', linewidth=0.5, linestyle='--',
            label='Random Classifier')
    ax.legend(loc="lower right")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('{} Receiver Operator Characteristic curves'.format(dev))
    sns.despine()
    plt.show()


# Write ROC AUC scores to csv
os.makedirs(os.path.abspath('evaluations'), exist_ok=True)
pd.DataFrame(roc_auc_scores).to_csv(os.path.abspath('evaluations/roc_auc_scores.csv'))



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



print("@@@ DONE @@@")
