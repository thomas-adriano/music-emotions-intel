import sys

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

# Print iterations progress
def print_progressbar(iteration, total, prefix='', suffix='', bottom="", decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    data = '\r%s |%s| %s%s %s\n%s' % (prefix, bar, percents, '%', suffix, bottom)

    data = CURSOR_UP_ONE + ERASE_LINE + data + '\n'

    sys.stdout.write(data)
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
