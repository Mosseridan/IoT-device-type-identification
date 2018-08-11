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
from sklearn.feature_selection import chi2, SelectKBest

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

from device_session_regressor import DeviceSessionRegressor
from device_session_classifier import DeviceSessionClassifier
from device_sequence_classifier import DeviceSequenceClassifier


def is_dev(this_dev_name, dev_name):
    return 1 if this_dev_name == dev_name else 0


def get_is_dev_vec(this_dev_name, dev_names):
    """
    This method generates a list with entries 0 or 1 to indicate which of the
    entries in the dev_names list is the device we are currently training/testing
    a classifier for. 
    """
    return [is_dev(this_dev_name, dev_name) for dev_name in dev_names]


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

def perform_feature_selection(X_train, y_train, k_val):
    """ This method is used in order to perform a feature selection by selecting
    the best k_val features from X_train. It does so according to the chi2
    criterion. The method prints the chosen features and creates
    a new instance of X_train with only these features and returns it 
    """
    print("**********FEATURE SELECTION**********")
    # Create and fit selector
    selector = SelectKBest(chi2, k=k_val)
    selector.fit(X_train, y_train)
    #Get idxs of columns to keep
    idxs_selected = selector.get_support(indices=True)
    print(idxs_selected)
    X_new = SelectKBest(chi2, k=k_val).fit_transform(X_train, y_train)
    return X_new
    

def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    score = roc_auc_score(y_true, y_proba[:, 1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)" % (label, score))
    return score



y_col = 'device_category'
cols_to_drop = ['device_category']
use_cols = pd.read_csv(os.path.abspath('data/use_cols.csv'))

# Load Data
train = pd.read_csv(os.path.abspath('data/train.csv'), usecols=use_cols, low_memory=False)
validation = pd.read_csv(os.path.abspath('data/validation.csv'), usecols=use_cols, low_memory=False)
test = pd.read_csv(os.path.abspath('data/test.csv'), usecols=use_cols, low_memory=False)

# Get device list
devices = train[y_col].unique()

# Shuffle Data
train = shuffle(train)
validation = shuffle(validation)
test = shuffle(test)

# Remove missing data
train = clear_missing_data(train)
validation = clear_missing_data(validation)
test = clear_missing_data(test)

# Seperate data to features
x_train = train.drop(cols_to_drop, 1)
x_validation = validation.drop(cols_to_drop, 1)
x_test = test.drop(cols_to_drop, 1)

# Perform feature scaling for X
x_train = perform_feature_scaling(x_train)
x_validation = perform_feature_scaling(x_validation)
x_test = perform_feature_scaling(x_test)

test_roc_auc_scores = {}
validation_roc_auc_scores = {}

for dev in devices:
    print("Learning models for device: {}".format(dev))

    # Get data labels
    y_train = pd.Series(get_is_dev_vec(dev, train[y_col]))
    y_validation = pd.Series(get_is_dev_vec(dev, validation[y_col]))
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

    n_splits = 10
    ub = BaggingClassifier(warm_start=True, n_estimators=0)
    for split in range(n_splits):
        x_res, y_res = RandomUnderSampler(random_state=split).fit_sample(x_train, y_train)
        ub.n_estimators += 1
        ub.fit(x_res, y_res)

    # Plot AUC ROC curves for validation data
    if y_test.sum() == 0:
        print('No {} sessions in validation data. skipping model evaluation.'.format(dev))
        continue

    f, ax = plt.subplots(figsize=(6, 6))

    validation_roc_auc_scores[dev] = [
        roc_auc_plot(y_validation, ub.predict_proba(x_validation), label='UB ', l='-'),
        roc_auc_plot(y_validation, forest.predict_proba(x_validation), label='FOREST ', l='--'),
        roc_auc_plot(y_validation, cart.predict_proba(x_validation), label='CART', l='-.'),
        roc_auc_plot(y_validation, rus.predict_proba(x_validation), label='RUS', l=':')
    ]

    ax.plot([0, 1], [0, 1], color='k', linewidth=0.5, linestyle='--',
            label='Random Classifier')
    ax.legend(loc="lower right")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('{} Validation  Receiver Operator Characteristic curves'.format(dev))
    sns.despine()
    plt.show()

    # Plot ROC AUC curves for test data
    if y_test.sum() == 0:
        print('No {} Test sessions in test data. skipping model evaluation.'.format(dev))
        continue

    f, ax = plt.subplots(figsize=(6, 6))

    test_roc_auc_scores[dev] = [
        roc_auc_plot(y_test, ub.predict_proba(x_test), label='UB ', l='-'),
        roc_auc_plot(y_test, forest.predict_proba(x_test), label='FOREST ', l='--'),
        roc_auc_plot(y_test, cart.predict_proba(x_test), label='CART', l='-.'),
        roc_auc_plot(y_test, rus.predict_proba(x_test), label='RUS', l=':')
    ]

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
pd.DataFrame(validation_roc_auc_scores).to_csv(os.path.abspath('evaluations/validation_roc_auc_scores.csv'))
pd.DataFrame(test_roc_auc_scores).to_csv(os.path.abspath('evaluations/test_roc_auc_scores.csv'))



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
