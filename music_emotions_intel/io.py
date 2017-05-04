import genericpath
import logging as log
import os
import pickle
import time
from collections import deque
from itertools import chain
from sys import getsizeof, stderr

import audioread
import librosa
import pandas as pd

from music_emotions_intel import configurations, feature_extraction, utils

AUDIO_DATA_NAME = 'audio_data'


class Audio_Data:
    def __init__(self, id_, audio_time_series, sr, tempo):
        self.id_ = int(id_)
        self.audio_time_series = audio_time_series
        self.sr = int(sr)
        self.tempo = tempo

    def __str__(self):
        return """
        id: %d
        y: %s
        sr: %d
        tempo: %d
        """ % (self.id_, str(self.audio_time_series), self.sr, self.tempo)

    def __sizeof__(self):
        return total_size(self.audio_time_series)


def load_annotations(annotations_path=configurations.DATASET_Y_PATH):
    log.debug('Loading annotations from path %s' % annotations_path)
    annotations = pd.read_csv(filepath_or_buffer=annotations_path, delimiter=',')
    res = pd.DataFrame(data={'ID': annotations['song_id'], 'AROUSAL': annotations['mean_arousal'],
                             'VALENCE': annotations['mean_valence']}, columns=['ID', 'AROUSAL', 'VALENCE'])
    res = res.set_index('ID')
    log.debug('Annotations successfully loaded')
    return res


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def file_iterator(path):
    for filename in os.listdir(path):
        if genericpath.isfile(os.path.join(path, filename)):
            yield path, filename


def load_audiodata_files(audio_cache_dir):
    log.debug('Loading optimized audio data file from %s' % audio_cache_dir)
    start_time = time.time()

    data = {}
    for path, filename in file_iterator(audio_cache_dir):
        if AUDIO_DATA_NAME not in filename:
            continue

        src = os.path.join(audio_cache_dir, filename)
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


def create_audiodata_files(audio_files, dest_dir, duration, sr, mono):
    log.debug('Creating optimized audio data file %s' % dest_dir)
    data = {}
    start_time = time.time()
    file_counter = 0
    size_threshold_bytes = 1500000000  # 1.5gb
    i = 0
    for f in audio_files:
        try:
            y, sr = load_audio(f, duration, sr, mono)
        except audioread.NoBackendError as e:
            log.warning('Impossible to load file %s as audio time series' % f)
            log.warning(e)
            continue
        id_ = utils.get_filename(f)
        tempo = feature_extraction.extract_tempo(y, sr)
        ad = Audio_Data(id_, y, sr, tempo)
        data[id_] = ad
        fname = AUDIO_DATA_NAME + '_' + str(file_counter)
        dest = os.path.join(dest_dir, fname)
        i += 1
        # workaround to MACOSX problems regarding pickling files above 1.5gb
        if (total_size(data) >= size_threshold_bytes) or (len(audio_files) == i):
            log.warning('writing %d audios' % len(data))
            with open(dest, 'w+b') as handle:
                file_counter += 1
                log.debug('Writing audio db file %s' % dest)
                pickle.dump(data, handle)
                handle.close()
            data = {}

    elapsed_time = time.time() - start_time
    log.info('Audio database successfully created in %d seconds' % elapsed_time)


def load_audio(path, duration_seconds=configurations.AUDIO_LENGTH_SECONDS, sr=configurations.AUDIO_SAMPLING_RATE, mono=True):
    log.debug('Loading audio time series from path %s' % path)
    audio_time_series, sr = librosa.load(path, sr=sr, mono=mono, duration=duration_seconds)
    log.debug('Audio time series successfully loaded')
    return audio_time_series, sr
