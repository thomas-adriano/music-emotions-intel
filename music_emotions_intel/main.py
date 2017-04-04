import operator

from sklearn import svm
import logging as log
from music_emotions_intel import feature_extraction as fe
from music_emotions_intel import io
import copy
log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

def load_data():
    annotations = io.load_annotations()
    attrs = fe.create_X()
    X = attrs.values
    return X, annotations


def slice_by_percentage(listlike, percent, from_=0):
    l = len(listlike)
    split = round(((percent * l) / 100) + 0.001)  # 0.001 force middle values like 2.5 be upped rounded
    if from_ != 0:
        split += from_
        if split > l:
            split = l
        return listlike[from_:split], join_matrices(listlike[:from_], listlike[split:])
    else:
        return listlike[from_:split], listlike[split:l]


def join_matrices(mat1, mat2):
    mat1 = copy.deepcopy(mat1).tolist()
    mat2 = copy.deepcopy(mat2).tolist()
    for v in mat2:
        mat1.append(v)
    return mat1


def index_for_composition(X, y_valence, y_arousal, percentage):
    l = len(X)
    i = round(l / ((percentage * l) / 100))
    Xs = []
    for ix in range(i):
        if ix == 0:
            prev_len = 0
        else:
            prev_len = len(Xs[0])
        Xs = slice_by_percentage(X, percentage, from_=operator.mul(ix, prev_len))
        ys_valence = slice_by_percentage(y_valence, percentage, from_=operator.mul(ix, prev_len))
        ys_arousal = slice_by_percentage(y_arousal, percentage, from_=operator.mul(ix, prev_len))
        yield {'X_test': Xs[0],
               'X_train': Xs[1],
               'y_valence_test': ys_valence[0],
               'y_valence_train': ys_valence[1],
               'y_arousal_test': ys_arousal[0],
               'y_arousal_train': ys_arousal[1]
               }


X, annotations = load_data()
y_arousal = annotations['AROUSAL']
y_valence = annotations['VALENCE']

y_hat_valence = io.load_binary('./resources/annotations/valence_yhat')
y_hat_arousal = io.load_binary('./resources/annotations/arousal_yhat')
# clf_valence = svm.SVR()
# clf_arousal = svm.SVR()
# for val in index_for_composition(X, y_valence, y_arousal, 25):
#     clf_valence.fit(val['X_train'], val['y_valence_train'])
#     y_hat_valence_part = clf_valence.predict(val['X_test']).tolist()
#     y_hat_valence = y_hat_valence + y_hat_valence_part
#
#     clf_arousal.fit(val['X_train'], val['y_arousal_train'])
#     y_hat_arousal_part = clf_arousal.predict(val['X_test']).tolist()
#     y_hat_arousal = y_hat_arousal + y_hat_arousal_part
#
# io.store(y_hat_valence,'./resources/annotations/valence_yhat')
# io.store(y_hat_arousal,'./resources/annotations/arousal_yhat')

print('ada')
# X, annotations = load_data()
# y_arousal = annotations['AROUSAL']
# y_valence = annotations['VALENCE']
#
#
#
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_arousal, test_size=0.33, random_state=42)
#
# clf = SVR()
# clf.fit(X_train, y_train)
# y_hat = clf.predict(X_test)
#
# mse = metrics.mean_squared_error(y_test, y_hat)
# r2 = metrics.r2_score(y_test, y_hat)
# evs = metrics.explained_variance_score(y_test, y_hat)
#
# print('MSE:', mse)
# print('r2:', r2)
# print('Explained Variance Score:', evs)

# MSE: 1.82064778259
# r2: -0.0119219966041
# Explained
# Variance
# Score: 0.0
