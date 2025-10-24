import logging

# Add a 'VERBOSE' logging level
logging.VERBOSE = logging.INFO - 1
logging.addLevelName(logging.VERBOSE, "VERBOSE")


#  Add a convenience method to the Logger class
def verbose(self, message, *args, **kws):
    if self.isEnabledFor(logging.VERBOSE):
        self._log(logging.VERBOSE, message, args, **kws)


logging.Logger.verbose = verbose


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def use_debugging_formatter():
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
