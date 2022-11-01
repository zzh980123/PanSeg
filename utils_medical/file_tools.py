import os


def create_dirs(fd) -> bool:
    if not os.path.exists(fd):
        os.makedirs(fd)
        return True
    return False
