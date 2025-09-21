import os

def is_main_process():
    return int(os.environ["RANK"]) == 0
