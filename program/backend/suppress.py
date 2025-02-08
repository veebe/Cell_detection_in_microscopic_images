import sys
import os

class SuppressStderr:
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._stderr