import os

dir_path = os.path.dirname(os.path.realpath(__file__))

AUDIO_CLIPS_PATH = os.path.join(os.sep, dir_path, 'resources/clips_45seconds')
STATIC_ANNOTATIONS_PATH = os.path.join(os.sep, dir_path, 'resources/static_annotations.csv')
DEFAULT_AUDIO_DB_DIR = os.path.join(os.sep, dir_path, 'resources/audio_db')