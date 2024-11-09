# forest

# Adding a 'VERBOSE' logging level
import logging
import haggis.logs
haggis.logs.add_logging_level("VERBOSE", logging.INFO - 1)

from .version import version
from . import bnbpy
