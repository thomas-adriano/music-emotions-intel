import logging as log

import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold

from music_emotions_intel import utils, cache
from music_emotions_intel.svmutil import *

utils.add_relative_file_path_to_sysenv(__file__)


def k_folds(X_df, y_df, folds=2):
    """
    Perform K folds in the provided X and y pandas dataframes
    :param X_df: pandas dataframe audio features with columns (ID, ft_1, ...ft_n) where ID column is the dataframe's index col
    :param y_df: pandas dataframe audio targets (valence and arousal) with columns (ID, VALENCE, AROUSAL) where ID column is the dataframe's index col
    :param folds: number of folds to perform
    :return: numpy ndarray of each of the following: ids of the testing fold, X of the training fold, 
    X of the test fold, y_valence of the training fold, y_arousal of the training fold,
    y_valence of the test fold, y_arousal of the test fold
    """
    X = cache.Xdf_to_Xndarray(X_df)
    y_valence, y_arousal = cache.ydf_to_yndarrays(y_df)
    ids = cache.get_Xdf_or_ydf_ids(y_df)
    kf = KFold(n_splits=folds)
    for train_index, test_index in kf.split(X):
        clf_valence = svm.SVR()
        clf_arousal = svm.SVR()
        yield ids[test_index], \
              X[train_index], X[test_index], \
              y_valence[train_index], y_arousal[train_index], \
              y_valence[test_index], y_arousal[test_index]


def train_classify(X_df, y_df, kfolds=5):
    """
    Train and classify using kfolds iteration and scikit learn svm.SVR algorithm
    :param X_df: pandas dataframe audio features with columns (ID, ft_1, ...ft_n) where ID column is the dataframe's index col
    :param y_df: pandas dataframe audio targets (valence and arousal) with columns (ID, VALENCE, AROUSAL) where ID column is the dataframe's index col
    :param kfolds: number of folds to perform the train/test
    :return: pandas dataframe containing the predicted arousal and valence with columns (ID, VALENCE, AROUSAL) where ID column is the dataframe's index col
    """
    df_list = []
    for idx, X_train, X_test, y_train_valence, y_train_arousal, y_test_valence, y_test_arousal in k_folds(X_df, y_df,
                                                                                                          kfolds):
        clf_valence = svm.SVR()
        clf_arousal = svm.SVR()
        clf_valence.fit(X_train, y_train_valence)
        clf_arousal.fit(X_train, y_train_arousal)

        df = cache.create_yhat_df(idx, clf_valence.predict(X_test).ravel(), clf_arousal.predict(X_test).ravel())
        df_list.append(df)
    res = pd.concat(df_list)
    return res


def train_classify_libsvm_svr(X_df, y_df, kfolds=5):
    """
    Train and classify using kfolds iteration and libsvm SVR
    :param X_df: pandas dataframe audio features with columns (ID, ft_1, ...ft_n) where ID column is the dataframe's index col
    :param y_df: pandas dataframe audio targets (valence and arousal) with columns (ID, VALENCE, AROUSAL) where ID column is the dataframe's index col
    :param kfolds: number of folds to perform the train/test
    :return: pandas dataframe containing the predicted arousal and valence with columns (ID, VALENCE, AROUSAL) where ID column is the dataframe's index col
    """
    df_list = []
    for idx, X_train, X_test, y_train_valence, y_train_arousal, y_test_valence, y_test_arousal in k_folds(X_df, y_df,
                                                                                                          kfolds):
        clf_valence = svm_train(y_train_valence.tolist(), X_train.tolist(), '-s 3')
        clf_arousal = svm_train(y_train_arousal.tolist(), X_train.tolist(), '-s 3')

        p_label_valence, p_val_valence = svm_predict(X_test.tolist(), clf_valence)
        p_label_arousal, p_val_arousal = svm_predict(X_test.tolist(), clf_arousal)

        df = cache.create_yhat_df(idx, p_label_valence, p_label_arousal)
        df_list.append(df)
    res = pd.concat(df_list)
    return res


def train_classify_tensorflow_nn(X_df, y_df, kfolds=5):
    """
    Train and classify using kfolds iteration and libsvm SVR
    :param X_df: pandas dataframe audio features with columns (ID, ft_1, ...ft_n) where ID column is the dataframe's index col
    :param y_df: pandas dataframe audio targets (valence and arousal) with columns (ID, VALENCE, AROUSAL) where ID column is the dataframe's index col
    :param kfolds: number of folds to perform the train/test
    :return: pandas dataframe containing the predicted arousal and valence with columns (ID, VALENCE, AROUSAL) where ID column is the dataframe's index col
    """
    df_list = []
    for idx, X_train, X_test, y_train_valence, y_train_arousal, y_test_valence, y_test_arousal in k_folds(X_df, y_df,
                                                                                                          kfolds):
        print('noop')
    res = pd.concat(df_list)
    return res


def train_using_cache():
    log.info('running model training using cached data...')
    X = cache.load_X_cache()
    y = cache.load_y_cache()
    print(train_classify(X, y))
    print(train_classify_libsvm_svr(X, y))
    log.info('training using cached data successfully finished!')


def train():
    log.info('running model training using source data...')


if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    train_using_cache()
