
"""
LWA Advanced Digital Processor control library
"""

__version__    = "0.1"
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2015, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

import MCS2
import Adp

# Internal modules for testing, debugging etc.
from ThreadPool import ThreadPool
from AdpRoach   import AdpRoach
