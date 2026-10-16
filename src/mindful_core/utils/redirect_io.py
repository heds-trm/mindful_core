import os
import sys
from contextlib import contextmanager


def _redirect_stdout(to, fd: int):
    sys.stdout.close()
    os.dup2(to.fileno(), fd)
    sys.stdout = os.fdopen(fd, "w")


def _redirect_stderr(to, fd: int):
    sys.stderr.close()
    os.dup2(to.fileno(), fd)
    sys.stderr = os.fdopen(fd, "w")


@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file, fd=fd)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout, fd=fd)  # restore stdout.


@contextmanager
def stderr_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stderr(to=file, fd=fd)
        try:
            yield
        finally:
            _redirect_stderr(to=old_stdout, fd=fd)
