import sys
from threading import Thread, Lock


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cprint(color, *args, sep=" ", end="\n", flush=False):
    """
    Prints a colored messange on the console
    :param color:           Color to display. Use bcolors from this class for a list of available colors
    :param args:            What to print, in standard python print syntax
    :param flush:           Print argument. Please refer to Python builtin print documentation
    :param file:            Print argument. Please refer to Python builtin print documentation
    :param end:             Print argument. Please refer to Python builtin print documentation
    :param sep:             Print argument. Please refer to Python builtin print documentation
    :return:
    """
    print(color, *args, bcolors.ENDC, sep=sep, end=end, flush=flush)


def fprint(path, *args):
    """
    Prints to a file using the standard print function. Useful to redirect console output to a file.
    This function is thread save using python Lock, to avoid the merge of two files and ensure thread-safety
        :param path:        The path where to write the file to
        :param args:        What to print, in standard python print syntax
    """
    mutex = Lock()
    mutex.acquire()

    original_stdout = sys.stdout
    with open(path, 'a+') as f:
        sys.stdout = f                                      # Change the standard output to the file we created.
        print(*args)
        sys.stdout = original_stdout                        # Reset the standard output to its original value
    mutex.release()
