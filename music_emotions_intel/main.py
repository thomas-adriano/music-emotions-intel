from sklearn import model_selection, metrics
from sklearn.svm import SVR
from music_emotions_intel import feature_extraction as fe

from music_emotions_intel import io, paths
dat = io.load_optimized_audio_data_file(paths.DEFAULT_AUDIO_DB_DIR)

for m in dat:
    print(dat[m])
# annotations = Annotations()
# attrs = fe.transform_features()
#
# X = attrs.values
# y = annotations.get_mean_arousals().values
#
# print('X shape', X.shape)
# print('y shape', y.shape)
#
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
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
# Explained Variance Score: 0.0
