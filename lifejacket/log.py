'''
Version: 1.2
Function: Log System
'''

import logging
from logging import handlers

format_dict = {
    1: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    2: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    3: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    4: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    5: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
}

class Logger():
    def __init__(self, logname, loglevel, logger):
        # create logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # create handler for writing file
        fh = handlers.RotatingFileHandler(filename=logname, mode='a', maxBytes=500*1024*1024, backupCount=3, encoding='UTF-8')
        fh.setLevel(logging.DEBUG)

        # create handler for console
        #ch = logging.StreamHandler()
        #ch.setLevel(logging.DEBUG)

        # handler output format
        formatter = format_dict[int(loglevel)]
        fh.setFormatter(formatter)
        #ch.setFormatter(formatter)

        # add handler
        self.logger.addHandler(fh)
        #self.logger.addHandler(ch)

    def getlog(self):
        return self.logger
