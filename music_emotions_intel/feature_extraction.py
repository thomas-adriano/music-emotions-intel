"""TODO: search in articles the recommended features to be extracted.
Put in the documentation of each module' function why this feature is necessary"""
import librosa as lr
import numpy as np
import pandas as pd
import sklearn

import music_emotions_intel.cache
from music_emotions_intel import io
from music_emotions_intel import configs
from music_emotions_intel import progress_bar


def create_X(audio_db_dir=configs.CACHE_AUDIO_CLIPS, sr=configs.DEFAULT_SAMPLING_RATE):
    print('extracting features...')
    res = []
    song_ids = []
    i = 0
    audio_data = music_emotions_intel.cache.load_audio_files_cache(audio_db_dir)
    cols = []
    for id_ in audio_data:
        audio_time_series = audio_data[id_].audio_time_series
        sr = audio_data[id_].sr
        tempo_ = audio_data[id_].tempo
        progress_bar.print_progressbar(i, len(audio_data), prefix='Progress:', suffix='Complete', bottom=id_,
                                       bar_length=50)

        y, mfcc_cols = extract_mfcc(audio_time_series, id_, sr)
        y = transform_mfcc(y)
        cols = ['mfcc_' + str(x) for x in range(len(y))]

        cols.append('tempo')
        y.append(tempo_)
        song_ids.append(id_)

        res.append(y)
        i += 1

    res = pd.DataFrame(res, columns=cols, index=song_ids)
    res = res.fillna(0)
    return res


def transform_mfcc(mfcc):
    # Support Vector Machines assume that all features are centered around zero and have variance in the same order
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    # reduce the array dimensionality to match the double audio duration in seconds
    mfcc = sklearn.decomposition.PCA(n_components=90).fit_transform(mfcc)
    # transform to 1d array
    mfcc = mfcc.ravel()
    return mfcc.tolist()


def extract_mfcc(y, name, sr=configs.DEFAULT_SAMPLING_RATE):
    """
    Compute the mel-frequency cepstral coefficients
    """
    mfcc = lr.feature.mfcc(y=y, sr=sr)
    columns = ['mfcc_' + str(x) for x in range(len(mfcc[0]))]
    return mfcc, columns


def tempo(y, sr):
    onset_env = lr.onset.onset_strength(y, sr=sr)
    tempo = lr.beat.estimate_tempo(onset_env, sr=sr)
    return tempo


def tempogram(y, name, sr=configs.DEFAULT_SAMPLING_RATE):
    tempogram = lr.feature.tempogram(y=y, sr=sr)
    columns = ['tempogram_' + str(x) for x in range(len(tempogram[0]))]
    return pd.DataFrame(tempogram, columns=columns)
