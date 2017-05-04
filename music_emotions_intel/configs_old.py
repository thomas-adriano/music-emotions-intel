import os

from music_emotions_intel import utils

dir_path = utils.get_file_parent_dir(__file__)
AUDIO_LENGTH_SECONDS = 45
AUDIO_MONO = True
AUDIO_SAMPLING_RATE = 44100

DATASET_AUDIOS_DIR = os.path.join(os.sep, dir_path, 'resources/clips_45seconds')
DATASET_Y_PATH = os.path.join(os.sep, dir_path, 'resources/static_annotations.csv')

MODEL_ALGORITHM = 'scikit.svm'
MODEL_NAME = 'default_model'

# constantes de configs
CACHE_Y_FILENAME = 'y.csv'
CACHE_YHAT_FILENAME = 'y_hat.csv'
CACHE_X_FILENAME = 'X.csv'
CACHE_AUDIO_FILES_DIR = os.path.join(os.sep, dir_path, 'cache_dir/audio/%(duration)s/%(samplingrate)s')
CACHE_TRAINING_DATA = os.path.join(os.sep, dir_path, 'cache_dir/training_data/%(duration)s/%(alg)s/%(feature_set)s')

FEATURE_EXTRACTION = {
    'extract_mfcc': True,
    'extract_spectral_contrast': True,
    'extract_chromagram': True,
    'extract_tempo': True,

    'n_mfcc_bands': 20,
    'n_spectral_contrast_bands': 6,

    'transform_mfcc': True,
    'transform_spectral_contrast': True,
    'transform_chromagram': True
}



DATABASE_INITIALIZATION_SCRIPT = os.path.join(os.sep, dir_path, 'resources/sql/init.sql')


