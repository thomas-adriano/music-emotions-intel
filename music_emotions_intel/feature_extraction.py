"""TODO: search in articles the recommended features to be extracted.
Put in the documentation of each module' function why this feature is necessary"""
import librosa as lr
import numpy as np
import pandas as pd
import sklearn

from music_emotions_intel import configurations, progress_bar, io


class Extractor:
    def __init__(self, configs):
        self.configs = configs

    def create_X_from_cache_audio_files(self):
        audiofiles_cache_dir = self.configs.resolve_cache_audio_clips_path(self.configs.audio_length_in_seconds,
                                                                           self.configs.audio_sampling_rate)
        print('Extracting features from audiodata files located in %s...' % audiofiles_cache_dir)
        X = []
        song_ids = []
        i = 0
        audio_data = io.load_audiodata_files(audiofiles_cache_dir)
        cols = []
        for id_ in audio_data:
            audio_time_series = audio_data[id_].audio_time_series
            power_spectogram = self.extract_power_spectogram(audio_time_series)
            sr = audio_data[id_].sr
            tempo_ = audio_data[id_].tempo
            progress_bar.print_progressbar(i, len(audio_data), prefix='Progress:', suffix='Complete', bottom=id_,
                                           bar_length=50)

            features = []
            cols = []

            mfcc, mfcc_cols = self.process_feature_as_matrix(self.extract_mfcc, 'mfcc',
                                                             audio_time_series=audio_time_series, sr=sr)
            spectral_contrast, spectral_contrast_cols = self.process_feature_as_matrix(self.extract_spectral_contrast,
                                                                                       'spectral_contrast',
                                                                                       power_spectogram=power_spectogram,
                                                                                       sr=sr)
            chromagram, chromagram_cols = self.process_feature_as_matrix(self.extract_chromagram, 'chromagram',
                                                                         power_spectogram=power_spectogram,
                                                                         sr=sr)

            features += mfcc
            cols += mfcc_cols

            features += spectral_contrast
            cols += spectral_contrast_cols

            features += chromagram
            cols += chromagram_cols

            if self.configs.feature_extract_tempo:
                # TEMPO
                features.append(tempo_)
                cols.append('tempo')

            # FINAL X FORM
            X.append(features)

            # NOT A FEATURE, JUST CREATING THE ID COLUMN
            song_ids.append(id_)
            i += 1

        X = pd.DataFrame(X, columns=cols)
        X = X.join(pd.DataFrame({'ID': song_ids}))
        X = X.set_index('ID')
        X = X.fillna(0)
        return X

    def process_feature_as_matrix(self, extract_fn, feature_name, audio_time_series=None, power_spectogram=None, sr=None):
        res = []
        cols = []
        if self.configs.as_dict()['FEATURE_EXTRACTION']['extract_' + feature_name]:
            extract_data = audio_time_series if audio_time_series != None else power_spectogram
            if extract_fn and extract_data is not None:
                features = extract_fn(extract_data, sr)
            else:
                raise ValueError("extract_fn and extract_data should not be None!")

            if self.configs.as_dict()['FEATURE_EXTRACTION']['transform_' + feature_name]:
                transformed_feature = self.transform_features(features, pca=True, normalize=True)
                res = transformed_feature
            else:
                res = self.ndarray_to_1dlist(features)
            cols = [feature_name + '_' + str(x) for x in range(len(res))]
        return res, cols

    def extract_power_spectogram(self, audio_time_series):
        return np.abs(lr.stft(audio_time_series))

    def extract_mfcc(self, audio_time_series, sr):
        """
        Compute the mel-frequency cepstral coefficients
        """
        mfcc = lr.feature.mfcc(y=audio_time_series, sr=sr, n_mfcc=configurations.FEATURE_EXTRACTION['n_mfcc_bands'])
        return mfcc

    def extract_spectral_contrast(self, power_spectogram, sr):
        contrast = lr.feature.spectral_contrast(S=power_spectogram, sr=sr,
                                                n_bands=configurations.FEATURE_EXTRACTION[
                                                    'n_spectral_contrast_bands'])
        return contrast

    def extract_chromagram(self, power_spectogram, sr):
        chroma = lr.feature.chroma_stft(S=power_spectogram, sr=sr)
        return chroma

    def extract_tempo(self, audio_time_series, sr):
        onset_env = lr.onset.onset_strength(audio_time_series, sr=sr)
        tempo = lr.beat.estimate_tempo(onset_env, sr=sr)
        return tempo

    def extract_tempogram(self, audio_time_series, sr):
        tempogram = lr.feature.tempogram(y=audio_time_series, sr=sr)
        return tempogram

    def transform_features(self, X, pca=True, pca_numcomponents=180, normalize=True):
        # Support Vector Machines assume that all features are centered around zero and have variance in the same order
        if normalize:
            X = self.apply_scaling(X)
        if pca:
            # reduce the array dimensionality to match the double audio duration in seconds
            X = self.apply_pca(X, pca_numcomponents)
        # transform to 1d array
        return self.ndarray_to_1dlist(X)

    def ndarray_to_1dlist(self, ndarr):
        return ndarr.ravel().tolist()

    def apply_pca(self, X, n_components):
        return sklearn.decomposition.PCA(n_components=n_components).fit_transform(X)

    def apply_scaling(self, X):
        return sklearn.preprocessing.scale(X, axis=1)
