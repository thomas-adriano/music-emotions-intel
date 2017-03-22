"""TODO: search in articles the recommended features to be extracted.
Put in the documentation of each module' function why this feature is necessary"""
import librosa as lr
import numpy as np
import pandas as pd
import sklearn

from music_emotions_intel import io
from music_emotions_intel import paths
from music_emotions_intel import progress_bar


def transform_features(sr=io.DEFAULT_SAMPLING_RATE):
    # print('extracting features...')
    res = np.array([])
    song_ids = []
    i = 0
    for (y, sr), dir_, name in io.load_audio_from_dir(paths.AUDIO_CLIPS_PATH, sr=sr, duration=45):
        # print('file', f)
        progress_bar.print_progressbar(i, len(y), prefix='Progress:', suffix='Complete', bottom=name,
                                       bar_length=50)
        song_ids.append(name)
        mfcc, mfcc_cols = extract_mfcc(y, name, sr)
        mfcc = transform_mfcc(mfcc)
        cols = np.array([])
        mfcc_cols = ['mfcc_' + str(x) for x in range(len(mfcc))]
        cols = np.concatenate((cols, mfcc_cols))
        res = np.concatenate((res, mfcc))

        tempo_ = tempo(y, sr)
        cols = np.append(cols, 'tempo')
        res = np.append(res, tempo_)

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
    return mfcc


def extract_mfcc(y, name, sr=io.DEFAULT_SAMPLING_RATE):
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


def tempogram(y, name, sr=io.DEFAULT_SAMPLING_RATE):
    tempogram = lr.feature.tempogram(y=y, sr=sr)
    columns = ['tempogram_' + str(x) for x in range(len(tempogram[0]))]
    return pd.DataFrame(tempogram, columns=columns)
