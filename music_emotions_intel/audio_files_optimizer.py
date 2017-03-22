from music_emotions_intel import io, paths
import os
import logging as log

log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

audio_files = [os.path.join(x[0], x[1]) for x in io.file_iterator(paths.AUDIO_CLIPS_PATH)]
io.create_optimized_audio_data_file(audio_files, duration=45)