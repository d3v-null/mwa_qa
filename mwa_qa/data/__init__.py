# init file for mwa_clysis/data
import os

DEFAULT_DATA_PATH = __path__[0]
DATA_PATH = os.environ.get("MWA_QA_DATA_PATH", DEFAULT_DATA_PATH)
