
from __future__ import print_function

import os
try:
    import simplejson as json
except ImportError:
    print("Warning: Failed to import simplejson; falling back to vanilla json")
    import json

def parse_config_file(filename, log=None):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def parse_config_file_old(filename, log=None):
    """
    Given a filename of a DP configuation file, read in the various values
    and return the requested configuation as a dictionary.
    """
    
    # Deal with logging
    #logger = logging.getLogger(__name__)
    if log is not None: log.info("Parsing config file '%s'", filename)

    # List of the required parameters and their coercion functions
    coerceMap = {'APPLICATIONDIR'           : str,
                 'BAMFILE'                  : str,
                 'ANTENNASTATFILE'          : str,
                 'ANTENNASTATPERIOD'        : int,
                 'DP1FIRMWAREFILENAME'      : str,
                 'DP2FIRMWAREFILENAME'      : str,
                 'MESSAGEHOST'              : str,
                 'MESSAGEOUTPORT'           : int,
                 'MESSAGEINPORT'            : int,
                 'TBNDATARECORDERHOST'      : str,
                 'TBNDATARECORDEROUTPORT'   : int,
                 'BEAM1DATARECORDERIP'      : str,
                 'BEAM1DATARECORDEROUTPORT' : int,
                 'BEAM1SRCIP'               : str,
                 'BEAM2DATARECORDERIP'      : str,
                 'BEAM2DATARECORDEROUTPORT' : int,
                 'BEAM2SRCIP'               : str,
                 'BEAM3DATARECORDERIP'      : str,
                 'BEAM3DATARECORDEROUTPORT' : int,
                 'BEAM3SRCIP'               : str, 
                 'BEAM4DATARECORDERIP'      : str,
                 'BEAM4DATARECORDEROUTPORT' : int,
                 'BEAM4SRCIP'               : str,
                 'DP2_BEAMS_BOARDS'         : eval,
                 'DP2_BEAMS_DRX_XILIDS'     : eval,
                 'DP2_BEAMS_ETH_XILIDS'     : eval,
                 'APPLICATIONDIR'           : str,
                 'BOARDCONFIGFILENAME'      : str,
                 'BOARDLOGFILENAME'         : str,
                 'MONITORPERIOD'            : int,
                 'TEMPERATUREWARN'          : int,
                 'TEMPERATURESHUTDOWN'      : int,
                 'VCCINTMIN'                : float,
                 'VCCINTMAX'                : float,
                 'VCCAUXMIN'                : float,
                 'VCCAUXMAX'                : float,
                 'DP_POWERMAX'              : int,
                 'FST_DEFAULT_COEFFS'       : str}
    config = {}

    #
    # read defaults config file
    #
    if not os.path.exists(filename):
        if log is not None: log.critical('Config file does not exist: %s', filename)
        sys.exit(1)

    cfile_error = False
    fh = open(filename, 'r')
    for line in fh:
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue    # ignore blank or comment line
            
        tokens = line.split()
        if len(tokens) != 2:
            if log is not None: log.error('Invalid config line, needs parameter-name and value: %s', line)
            cfile_error = True
            continue
        
        paramName = tokens[0].upper()
        if paramName in coerceMap.keys():
            # Try to do the type conversion and, for int's and float's, make sure
            # the values are greater than zero.
            try:
                val = coerceMap[paramName](tokens[1])
                if coerceMap[paramName] == int or coerceMap[paramName] == float:
                    if val <= 0:
                        if log is not None: log.error("Integer and float values must be > 0")
                        cfile_error = True
                    else:
                        config[paramName] = val
                else:
                    config[paramName] = val
                    
            except Exception as e:
                if log is not None: log.error("Error parsing parameter %s: %s", paramName, str(e))
                cfile_error = True
                
        else:
            if log is not None: log.warning("Unknown config parameter %s", paramName)
            
    # Verify that all required parameters were found
    for paramName in coerceMap.keys():
        if not paramName in config:
            if log is not None: log.error("Config parameter '%s' is missing", paramName)
            cfile_error = True
    if cfile_error:
        if log is not None: log.critical("Error parsing configuation file '%s'", filename)
        sys.exit(1)

    return config
