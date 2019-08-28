
import logging
from logging.handlers import TimedRotatingFileHandler
import time
import os

class AdpFileHandler(TimedRotatingFileHandler):
    def __init__(self, config, filename, rollover_callback=None):
        days_per_file = config['log']['days_per_file']
        file_count    = config['log']['max_file_count']
        TimedRotatingFileHandler.__init__(self, filename, when='D',
                                          interval=days_per_file,
                                          backupCount=file_count)
        self.filename = filename
        self.rollover_callback = rollover_callback
    def doRollover(self):
        super(AdpFileHandler, self).doRollover()
        if self.rollover_callback is not None:
            self.rollover_callback()

class AdpFileLogger(object):
    def __init__(self, config, filename, fileheader, *args, **kwargs):
        fmt     = config['log']['stats_format']
        datefmt = config['log']['date_format']
        self._fileheader = fileheader
        self._formatter = logging.Formatter(fmt, datefmt=datefmt)
        self._formatter.converter = time.gmtime
        self._handler = AdpFileHandler(config, filename,
                                       rollover_callback=self.on_rollover,
                                       *args, **kwargs)
        self._handler.setFormatter(self._formatter)
        self._log     = logging.getLogger(filename)
        self._log.level = logging.INFO
        self._log.addHandler(self._handler)
        if os.path.getsize(filename) == 0:
            self.on_rollover()
    def on_rollover(self):
        for line in self._fileheader:
            self.log(line)
    def log(self, *args, **kwargs):
        self._log.info(*args, **kwargs)
