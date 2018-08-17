import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
import seaborn as sns

sns.set_style("white")

def load_model_from_pkl(pkl):
    return joblib.load(os.path.abspath(pkl))


def load_data_from_csv(csv, use_cols=None):
    return pd.read_csv(os.path.abspath(csv), usecols=use_cols, low_memory=False)


def perform_feature_scaling(features):
    """
    This method is used in order to perform feature scaling according to the
    min-max scaler. The scaler can be replaced with another one, like the
    standard scaler
    """
    scaler = MinMaxScaler()
    scaler.fit(features)
    return pd.DataFrame(scaler.transform(features), columns=features.columns)


def split_data(data, y_col):
    # Remove missing data
    data = clear_missing_data(data)
    x = perform_feature_scaling(data.drop([y_col], axis=1))
    y = data[y_col]
    return x, y


def clear_missing_data(df):
    df_with_nan = df.replace("?", np.NaN)
    return df_with_nan.dropna(0)


def take_n_of_each(df, n):
    return df.groupby('device_category').head(n)


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
    # Get idxs of columns to keep
    idxs_selected = selector.get_support(indices=True)
    print(idxs_selected)
    x_new = SelectKBest(chi2, k=k_val).fit_transform(X_train, y_train)
    return x_new


def roc_auc_plot(y_true, y_proba, ax, label=' ', l='-', lw=1.0):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    score = metrics.roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)" % (label, score))
    return score


def eval_predictions(y_true, y_pred):
    # Accuracy classification score.
    accuracy_score = metrics.accuracy_score(y_true, y_pred)
    # Build a text report showing the main classification metrics
    classification_report = metrics.classification_report(y_true, y_pred)
    # Compute confusion matrix to evaluate the accuracy of a classification
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.ravel()
    # Compute precision, recall, F-measure and support for each class
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    roc_auc_score = metrics.roc_auc_score(y_true, y_pred)
    print(classification_report)

    # classes = sorted(list(pd.Series(y_true).unique()))
    classes = [0,1]
    return {
        'class': classes,
        'accuracy_score': [accuracy_score]*len(classes),
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'support': support,
        'roc_auc_score': [roc_auc_score]*len(classes),
        'tp': [tp]*len(classes),
        'fp': [fp]*len(classes),
        'tn': [tn]*len(classes),
        'fn': [fn]*len(classes)
    }
