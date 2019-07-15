""" utils.py

    This module sets up a logfile using the current time
"""

import logging
import time

time_s = time.strftime("%d_%b_%H_%M_%S")
fname = "logs/{}".format(time_s)

print("Logfile created: {}".format(fname))

fh = logging.FileHandler(fname)
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

# create formatter and add it to the handlers
formatstr = '%(asctime)s: %(name)s: %(levelname)s: %(message)s'
formatter = logging.Formatter(formatstr, datefmt='%m\%d %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
