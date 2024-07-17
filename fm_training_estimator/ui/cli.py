# Standard
import logging

# Third Party
import fire

# Local
from .core import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(run)
