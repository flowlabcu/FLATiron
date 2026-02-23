import sys 

def custom_warning_message(warning_msg):
    """
    Print a warning message to the terminal with a standard format.
    """
    print(f'\033[91m[WARNING]\033[0m {warning_msg}')

def debug_print(*args, **kwargs):
    """
    Debug print function that only prints if the script is run with the
    --debug flag.
    """
    if '--debug' in sys.argv:
        print('\033[92m[DEBUG]\033[0m', *args, **kwargs)