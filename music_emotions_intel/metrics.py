import logging as log
import time

import sklearn as skl

from music_emotions_intel import cache, configurations, db


class Meter:
    def __init__(self, configs):
        self.configs = configs

    def persist_metrics(self, X, y, yhat, y_name):
        mse = skl.metrics.mean_squared_error(y, yhat)
        r2 = skl.metrics.r2_score(y, yhat)
        evs = skl.metrics.explained_variance_score(y, yhat)
        pval = skl.feature_selection.f_regression(X, yhat)
        pval_leq5 = len([x for x in pval[0] if x <= 0.05])
        num_features = len(X[0])

        payload = [("timestamp", time.time()), ("audio_length_seconds", self.configs.audio_length_in_seconds),
                   ("mean_squared_error", mse), ("r2", r2),
                   ("explained_variance_score", evs), ("num_features", num_features),
                   ("num_features_below_005_pval", pval_leq5), ("y_name", y_name),
                   ("algorithm", self.configs.model_algorithm),
                   ('audio_sampling_rate', self.configs.audio_sampling_rate),
                   ("model_name", self.configs.model_name)]

        log.debug("inserting col/values %s" % payload)

        db.insert("METRICS", payload)

    def perform_metrics_storage(self, X_df, y_df, yhat_df):
        self.persist_metrics(cache.Xdf_to_Xndarray(X_df),
                             cache.ydf_to_yndarrays(y_df)[0],
                             cache.ydf_to_yndarrays(yhat_df)[0], 'VALENCE')
        self.persist_metrics(cache.Xdf_to_Xndarray(X_df),
                             cache.ydf_to_yndarrays(y_df)[1],
                             cache.ydf_to_yndarrays(yhat_df)[1], 'AROUSAL')


if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    log.info('collecting metrics using cached data...')
    db.init_db()
    X = cache.load_X_cache(algorithm="svmlib", featureset="chromagram_360pca")
    y = cache.load_y_cache()
    yhat = cache.load_yhat_cache(algorithm="svmlib", featureset="chromagram_360pca")

    Meter(configurations.get_instance()).perform_metrics_storage(X, y, yhat)

    log.info('ceched data metrics successfully collected!')
