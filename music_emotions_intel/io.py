import logging as log
import os
import pickle
import tarfile
from collections import deque
from genericpath import isfile
from itertools import chain
from os.path import join
from sys import getsizeof, stderr
import pandas

import librosa
import pandas as pd

from music_emotions_intel import configs

AUDIO_DATA_NAME = 'audio_data'


def load_annotations(annotations_path=configs.STATIC_ANNOTATIONS_PATH):
    log.debug('Loading annotations from path %s' % annotations_path)
    annotations = pd.read_csv(filepath_or_buffer=annotations_path, delimiter=',')
    res = pd.DataFrame(data={'ID': annotations['song_id'], 'AROUSAL': annotations['mean_arousal'],
                             'VALENCE': annotations['mean_valence']}, columns=['ID', 'AROUSAL', 'VALENCE'])
    res = res.set_index('ID')
    log.debug('Annotations successfully loaded')
    return res


def load_audio_from_dir(path, duration=None, sr=configs.DEFAULT_SAMPLING_RATE, mono=True):
    log.debug('Loading audio files from path %s' % path)
    for dir_, f in file_iterator(path):
        y = load_audio(os.path.join(dir_, f), duration=duration, sr=sr, mono=mono)
        yield (y, dir_, f)


def load_audio(path, duration=None, sr=configs.DEFAULT_SAMPLING_RATE, mono=True):
    log.debug('Loading audio time series from path %s' % path)
    y, sr = librosa.load(path, sr=sr, mono=mono, duration=duration)
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


def file_iterator(path):
    for filename in os.listdir(path):
        if isfile(join(path, filename)):
            yield path, filename


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


def store_binary(obj, dest):
    with open(dest, 'w+b') as handle:
        log.debug('Writing object into %s' % dest)
        pickle.dump(obj, handle)
        handle.close()

def load_binary(src):
    with open(src, 'r+b') as handle:
        d = pickle.load(handle)
        handle.close()
    return d

def store_X_csv(data):
    pd.to_csv(pd.DataFrame(data, columns=columns, index=indexes))

def read_csv(src, index_col=None):
    df = pd.from_csv(src);
    if index_col:
        df = df.set_index(index_col)
    return df

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
