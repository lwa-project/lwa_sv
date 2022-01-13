# -*- coding: utf-8 -*-

"""
LWA Advanced Digital Processor control library
"""

from __future__ import absolute_import

__version__    = "0.1"
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2015, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

from . import MCS2
from . import Adp

# Internal modules for testing, debugging etc.
#from ThreadPool import ThreadPool
#from AdpRoach   import AdpRoach
from . import ThreadPool
from .AdpRoach import AdpRoach
