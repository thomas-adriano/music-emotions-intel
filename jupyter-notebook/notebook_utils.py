import os
import sys
import re

def prepare_notebook_path():
    cwd = os.getcwd()
    sys.path.append(cwd)

    cwd_arr = re.split('\\\\|/', cwd)
    cwd_arr = cwd_arr[:len(cwd_arr)-1]
    sys.path.append('/'.join(cwd_arr))

    cwd_arr.append('music_emotions_intel')
    sys.path.append('/'.join(cwd_arr))

