import os
import sys
import functools


def add_relative_file_path_to_sysenv(filepath):
    sys.path = [os.path.dirname(os.path.abspath(filepath))] + sys.path


def get_filename(path, exclude_extension=True):
    if exclude_extension:
        dot_index = path.rfind('.')
        slash_index = path.rfind('/')
        res = path[slash_index + 1:dot_index]
    else:
        slash_index = path.rfind('/')
        res = path[slash_index + 1:]
    return res


def create_dir_if_inexistent(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_file_contents(filepath):
    with open(filepath, 'r') as f:
        return f.read()


def get_file_parent_dir(filename):
    return os.path.dirname(os.path.realpath(filename))