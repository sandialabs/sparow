import logging
import haggis.logs

# Adding a 'VERBOSE' logging level
haggis.logs.add_logging_level("VERBOSE", logging.INFO - 1)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def use_debugging_formatter():
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
