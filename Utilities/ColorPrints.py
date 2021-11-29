import sys


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


def cprint(color, *args):
    """
    Prints a colored messange on the console
    :param color:           Color to display. Use bcolors from this class for a list of available colors
    :param args:            What to print, in standard python print syntax
    :return:
    """
    print(color, *args, bcolors.ENDC)


def fprint(path, *args):
    """
    Prints to a file using the standard print function. Useful to redirect console output to a file
        :param path:        The path where to write the file to
        :param args:        What to print, in standard python print syntax
    """
    original_stdout = sys.stdout
    with open(path, 'w+') as f:
        sys.stdout = f                                      # Change the standard output to the file we created.
        print(*args)
        sys.stdout = original_stdout                        # Reset the standard output to its original value
