"""Explicit isolated process entrypoint used by an Orin agent job."""
from .device import worker_job

if __name__ == "__main__":
    import sys
    worker_job(sys.argv[1])
