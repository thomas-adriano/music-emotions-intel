import logging as log
import os
import pickle
import time
from os.path import join
import pandas as pd
import audioread

from music_emotions_intel import io, configs, feature_extraction, training
from music_emotions_intel.io import file_iterator, AUDIO_DATA_NAME, load_audio, get_filename, Audio_Data, total_size

Y_CACHE_FILE = join(configs.CACHE_TRAINING_DATA, 'y')
YHAT_CACHE_FILE = join(configs.CACHE_TRAINING_DATA, 'y_hat_valence')
X_CACHE_FILE = join(configs.CACHE_TRAINING_DATA, 'X')


def create_audio_files_cache():
    audio_files = [os.path.join(x[0], x[1]) for x in io.file_iterator(configs.AUDIO_CLIPS_PATH)]
    _create_cache_audio_data_file(audio_files, duration=45)


def create_yhat_cache(yhats):
    yhats.to_csv(YHAT_CACHE_FILE)


def create_X_cache():
    X = feature_extraction.create_X()
    X.to_csv(X_CACHE_FILE)


def create_y_cache():
    annotations = io.load_annotations()
    annotations.to_csv(Y_CACHE_FILE)


def load_audio_files_cache(audio_cache_dir=configs.CACHE_AUDIO_CLIPS):
    log.debug('Loading optimized audio data file from %s' % audio_cache_dir)
    start_time = time.time()

    data = {}
    for path, filename in file_iterator(audio_cache_dir):
        if AUDIO_DATA_NAME not in filename:
            continue

        src = join(audio_cache_dir, filename)
        with open(src, 'r+b') as handle:
            try:
                d = pickle.load(handle)
                data.update(d)
                handle.close()
                elapsed_time = time.time() - start_time
                log.info('Optimized audio data file successfully loaded in %d seconds' % elapsed_time)
            except (EOFError, IOError) as e:
                log.error('Could not load file %s' % src)
                log.error(e)
                raise e

    return data


def load_y_cache():
    return io.load_binary(Y_CACHE_FILE)


def load_X_cache():
    return io.load_binary(X_CACHE_FILE)


def load_yhat_cache():
    return io.load_binary(YHAT_CACHE_FILE)


def _create_cache_audio_data_file(audio_files, duration=None, sr=configs.DEFAULT_SAMPLING_RATE, mono=True,
                                  audio_db_dir=configs.CACHE_AUDIO_CLIPS):
    log.debug('Creating optimized audio data file %s' % audio_db_dir)
    data = {}
    start_time = time.time()
    c2 = 0
    size_threshold_bytes = 1500000000  # 1.5gb
    i = 0
    for f in audio_files:
        try:
            y, sr = load_audio(f, duration, sr, mono)
        except audioread.NoBackendError as e:
            log.warning('Impossible to load file %s as audio time series' % f)
            log.warning(e)
            continue
        id_ = get_filename(f)
        tempo = feature_extraction.tempo(y, sr)
        ad = Audio_Data(id_, y, sr, tempo)
        data[id_] = ad
        fname = AUDIO_DATA_NAME + '_' + str(c2)
        dest = join(audio_db_dir, fname)
        i += 1
        # workaround to MACOSX problems regarding pickling files above 1.5gb
        if (total_size(data) >= size_threshold_bytes) or (len(audio_files) == i):
            log.warning('writing %d audios' % len(data))
            with open(dest, 'w+b') as handle:
                c2 += 1
                log.debug('Writing audio db file %s' % dest)
                pickle.dump(data, handle)
                handle.close()
            data = {}

    elapsed_time = time.time() - start_time
    log.info('Audio database successfully created in %d seconds' % elapsed_time)


if __name__ == '__main__':
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    # create_audio_files_cache()
    # create_y_cache()
    # create_X_cache()
    X = load_X_cache()
    y = load_y_cache()
    yhats = training.train_classify(X, y['VALENCE'], y['AROUSAL'])
    create_yhat_cache(yhats)
    yhat = load_yhat_cache()
    print('d')
