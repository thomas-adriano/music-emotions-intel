import logging as log
import os
import pickle
import tarfile
import time
from collections import deque
from genericpath import isfile
from itertools import chain
from os.path import join
from sys import getsizeof, stderr

import audioread
import librosa as lr
import pandas as pd

from music_emotions_intel import paths

DEFAULT_SAMPLING_RATE = 44100
AUDIO_DATA_NAME = 'audio_data'


def load_annotations(annotations_path=paths.STATIC_ANNOTATIONS_PATH):
    log.debug('Loading annotations from path %s' % annotations_path)
    annotations = pd.read_csv(filepath_or_buffer=annotations_path, delimiter=',')
    res = pd.DataFrame(data={'ID': annotations['song_id'], 'AROUSAL': annotations['mean_arousal'],
                             'VALENCE': annotations['mean_valence']}, columns=['ID', 'AROUSAL', 'VALENCE'])
    res = res.set_index('ID')
    log.debug('Annotations successfully loaded')
    return res


def load_audio_from_dir(path, duration=None, sr=DEFAULT_SAMPLING_RATE, mono=True):
    log.debug('Loading audio files from path %s' % path)
    for dir_, f in file_iterator(path):
        y = load_audio(os.path.join(dir_, f), duration=duration, sr=sr, mono=mono)
        yield (y, dir_, f)


def load_audio(path, duration=None, sr=DEFAULT_SAMPLING_RATE, mono=True):
    log.debug('Loading audio time series from path %s' % path)
    y, sr = lr.load(path, sr=sr, mono=mono, duration=duration)
    log.debug('Audio time series successfully loaded')
    return y, sr


def get_filename(path, exclude_extension=True):
    if exclude_extension:
        dot_index = path.rfind('.')
        slash_index = path.rfind('/')
        res = path[slash_index + 1:dot_index]
    else:
        slash_index = path.rfind('/')
        res = path[slash_index + 1:]
    return res


def create_optimized_audio_data_file(audio_files, duration=None, sr=DEFAULT_SAMPLING_RATE, mono=True,
                                     audio_db_dir=paths.DEFAULT_AUDIO_DB_DIR):
    log.debug('Creating optimized audio data file %s' % audio_db_dir)
    data = {}
    start_time = time.time()
    c2 = 0
    size_threshold = 1500000000
    i = 0
    for f in audio_files:
        # y, sr = []
        try:
            y, sr = load_audio(f, duration, sr, mono)
        except audioread.NoBackendError as e:
            log.warning('Impossible to load file %s as audio time series' % f)
            log.warning(e)
            continue
        id_ = get_filename(f)
        ad = Audio_Data(id_, y, sr)
        data[id_] = ad
        fname = AUDIO_DATA_NAME + '_' + str(c2)
        dest = join(audio_db_dir, fname)
        i += 1
        if (total_size(data) >= size_threshold) or (len(audio_files) - 1 == i):
            with open(dest, 'w+b') as handle:
                c2 += 1
                log.debug('Writing audio db file %s' % dest)
                pickle.dump(data, handle)
                handle.close()
            data = {}

    elapsed_time = time.time() - start_time
    log.info('Audio database successfully created in %d seconds' % (elapsed_time))


def load_optimized_audio_data_file(audio_db_dir=paths.DEFAULT_AUDIO_DB_DIR):
    log.debug('Loading optimized audio data file from %s' % audio_db_dir)
    start_time = time.time()

    data = {}
    for path, filename in file_iterator(audio_db_dir):
        if AUDIO_DATA_NAME not in filename:
            continue

        src = join(audio_db_dir, filename)
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


def file_iterator(path):
    for filename in os.listdir(path):
        if isfile(join(path, filename)):
            yield path, filename


class Audio_Data:
    def __init__(self, id_, y, sr):
        self.id_ = int(id_)
        self.y = y
        self.sr = int(sr)

    def __str__(self):
        return """
        id: %d
        y: %s
        sr: %d
        """ % (self.id_, str(self.y), self.sr)

    def __sizeof__(self):
        return total_size(self.y)


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


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
