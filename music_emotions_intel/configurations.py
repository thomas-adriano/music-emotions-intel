import configparser
import os

from music_emotions_intel import utils

DEFAULT_CONFIGS_FILE_PATH = ''

_DEFAULT_CONFIGS = configparser.ConfigParser()
_DEFAULT_CONFIGS['CACHE'] = {
    'audio_files_dir': 'audio/%(duration)s/%(samplingrate)s',
    'y_path': 'y/%(duration)s/y.csv',
    'yhat_path': 'yhat/%(duration)s/%(alg)s/%(model_name)s/y_hat.csv',
    'x_path': 'x/%(duration)s/%(alg)s/%(model_name)s/X.csv'
}

class Configs:
    def __init__(self):
        self.configs_values = self.load_config_file_if_inexistent()
        self.own_file_path = utils.get_file_parent_dir(__file__)

        self.dataset_audios_dir = self.configs_values['dataset']['audios_dir']
        self.dataset_y_path = self.configs_values['dataset']['y_path']

        self.audio_length_in_seconds = self.configs_values['audio']['length_in_seconds']
        self.audio_mono = self.configs_values['audio']['mono']
        self.audio_sampling_rate = self.configs_values['audio']['sampling_rate']

        self.model_algorithm = self.configs_values['model']['algorithm']
        self.model_name = self.configs_values['model']['name']

        self.feature_extraction_extract_mfcc = self.configs_values['feature_extraction']['extract_mfcc']
        self.feature_extract_spectral_contrast = self.configs_values['feature_extraction']['extract_spectral_contrast']
        self.feature_extract_chromagram = self.configs_values['feature_extraction']['extract_chromagram']
        self.feature_extract_tempo = self.configs_values['feature_extraction']['extract_tempo']
        self.feature_n_mfcc_bands = self.configs_values['feature_extraction']['n_mfcc_bands']
        self.feature_transform_mfcc = self.configs_values['feature_extraction']['transform_mfcc']
        self.feature_n_spectral_contrast_bands = self.configs_values['feature_extraction']['n_spectral_contrast_bands']
        self.feature_transform_spectral_contrast = self.configs_values['feature_extraction'][
            'transform_spectral_contrast']
        self.feature_transform_chromagram = self.configs_values['feature_extraction']['transform_chromagram']

        self.cache_dir = self.configs_values['cache']['dir']
        self.cache_audio_files_dir = self.configs_values['cache']['audio_files_dir']
        self.cache_y_path = self.configs_values['cache']['y_path']
        self.cache_yhat_path = self.configs_values['cache']['yhat_path']
        self.cache_x_path = self.configs_values['cache']['x_path']
        self.cache_create_audio_files_cache = self.configs_values['cache']['create_audio_files_cache']
        self.cache_create_y_cache = self.configs_values['cache']['create_y_cache']
        self.cache_create_x_cache = self.configs_values['cache']['create_x_cache']
        self.cache_create_yhat_cache = self.configs_values['cache']['create_yhat_cache']

        self.database_initialization_script_path = self.configs_values['database']['initialization_script_path']

    def load_config_file_if_inexistent(self):
        if self.configs_values:
            return

        cfg_parser = configparser.ConfigParser()
        cfg_parser.read_dict(_DEFAULT_CONFIGS)
        cfg_parser.read(self.own_file_path)
        return cfg_parser.items()

    def get_config(self, key):
        return self.configs_values[key]

    def contains(self, key):
        from_ini = self.get_config(key)
        from_ini_found = from_ini is not None and str(from_ini).strip()
        return from_ini_found

    def get_cache_audio_files_dir(self):
        return self.cache_audio_files_dir % {'duration': str(self.audio_length_in_seconds),
                                             'samplingrate': self.audio_sampling_rate}

    def get_cache_x_path(self):
        return self._prepend_own_path(
            self.cache_x_path % {'duration': str(self.audio_length_in_seconds), 'alg': self.model_algorithm,
                                 'model_name': self.model_name})

    def get_cache_y_path(self):
        return self._prepend_own_path(self.cache_y_path % {'duration': str(self.audio_length_in_seconds)})

    def get_cache_yhat_path(self):
        return self._prepend_own_path(
            self.cache_yhat_path % {'duration': str(self.audio_length_in_seconds),
                                    'alg': self.model_algorithm,
                                    'model_name': self.model_name})

    def as_dict(self):
        return self.configs_values

    def _prepend_own_path(self, str):
        return os.path.join(os.sep, self.own_file_path, str)

    def __str__(self):
        return str(self.configs_values)

_INSTANCE = Configs

def get_instance():
    if not _INSTANCE:
        _create_instance()
    return _INSTANCE


def _create_instance():
    _INSTANCE()
