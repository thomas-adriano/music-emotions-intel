import os

dir_path = os.path.dirname(os.path.realpath(__file__))

AUDIO_CLIPS_PATH = os.path.join(os.sep, dir_path, 'resources/clips_45seconds')
STATIC_ANNOTATIONS_PATH = os.path.join(os.sep, dir_path, 'resources/static_annotations.csv')
CACHE_AUDIO_CLIPS = os.path.join(os.sep, dir_path, 'cache/audio')
CACHE_TRAINING_DATA = os.path.join(os.sep, dir_path, 'cache/training_data')

DEFAULT_SAMPLING_RATE = 44100
