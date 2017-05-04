import logging as log
import os

import pandas as pd

from music_emotions_intel import io, feature_extraction, utils, configurations

if __name__ == '__main__':
    pass


class CacheHandler:
    def __init__(self, configs):
        self.configs

    def create_audio_files_cache(self):
        audio_files = [os.path.join(x[0], x[1]) for x in io.file_iterator(self.configs.dataset_audios_dir)]
        dest = self.configs.get_cache_audio_files_dir()
        utils.create_dir_if_inexistent(dest)
        io.create_audiodata_files(audio_files, dest, self.configs.audio_length_in_seconds,
                                  self.configs.audio_sampling_rate, self.configs.audio_mono)

    def create_X_cache(self):
        X = feature_extraction.create_X_from_cache_audio_files(self.configs.audio_length_in_seconds,
                                                               self.configs.audio_sampling_rate)

        X_path = self.configs.get_cache_x_path()
        dir = utils.get_file_parent_dir(X_path)
        utils.create_dir_if_inexistent(dir)
        X.to_csv(X_path)

    def create_y_cache(self):
        annotations = io.load_annotations()
        y_path = self.configs.get_cache_y_path()
        dir = utils.get_file_parent_dir(y_path)
        utils.create_dir_if_inexistent(dir)
        annotations.to_csv(y_path)

    def create_yhat_cache(self, yhats):
        yhat_path = self.configs.get_cache_yhat_path()
        dir = utils.get_file_parent_dir(yhat_path)
        utils.create_dir_if_inexistent(dir)
        yhats.to_csv(yhat_path)

    def load_y_cache(self):
        y_path = self.configs.get_cache_y_path()
        return pd.read_csv(y_path)

    def load_X_cache(self):
        X_path = self.configs.get_cache_x_path()
        return pd.read_csv(X_path)

    def load_yhat_cache(self):
        y_path = self.configs.get_cache_y_path()
        return pd.read_csv(y_path)

    def ydf_to_yndarrays(ydf):
        y_valence = ydf['AROUSAL'].as_matrix().ravel()
        y_arousal = ydf['VALENCE'].as_matrix().ravel()
        return y_valence, y_arousal

    def Xdf_to_Xndarray(Xdf):
        X = Xdf.drop('ID', axis=1).as_matrix()
        return X

    def get_Xdf_or_ydf_ids(df):
        ids = df['ID'].as_matrix().ravel()
        return ids

    def create_yhat_df(ids, yhat_valence, yhat_arousal):
        df = pd.DataFrame({'ID': ids, 'VALENCE': yhat_valence,
                           'AROUSAL': yhat_arousal})
        df = df.set_index('ID')
        return df

    def create_caches(self):
        log.info('Creating cache for X, y (v/a) and y_hat (v/a)...')
        if self.configs.cache_create_audio_files_cache:
            self.create_audio_files_cache()
        elif self.configs.cache_create_y_cache:
            self.create_y_cache()
        elif self.configs.cache_create_x_cache:
            self.create_X_cache()
        elif self.configs.cache_create_yhat_cache:
            X = self.load_X_cache()
            y = self.load_y_cache()
            yhats = self.training.train_classify(X, y)
            self.create_yhat_cache(yhats)

        log.info('Cache creation process finished')


if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    CacheHandler(configurations.get_instance()).create_caches()
