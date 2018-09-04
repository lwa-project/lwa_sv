"""
bifrost pipeline monitoring
"""

import os
import copy
import glob
import time

BIFROST_STATS_BASE_DIR = '/dev/shm/bifrost/'

# From bifrost so that we don't need to import it for the control software
def _multi_convert(value):
    """
    Function try and convert numerical values to numerical types.
    """

    try:
        value = int(value, 10)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    return value

# From bifrost so that we don't need to import it for the control software
def load_by_filename(filename):
    """
    Function to read in a ProcLog file and return the contents as a 
    dictionary.
    """

    contents = {}
    with open(filename, 'r') as fh:
        ## Read the file all at once to avoid problems but only after it has a size
        for attempt in xrange(5):
            if os.path.getsize(filename) != 0:
                break
            time.sleep(0.001)
        lines = fh.read()

        ## Loop through lines
        for line in lines.split('\n'):
            ### Parse the key : value pairs
            try:
                key, value = line.split(':', 1)
            except ValueError:
                continue

            ### Trim off excess whitespace
            key = key.strip().rstrip()
            value = value.strip().rstrip()

            ### Convert and save
            contents[key] = _multi_convert(value)
            
    # Done
    return contents

# From bifrost so that we don't need to import it for the control software
def load_by_pid(pid, include_rings=False):
    """
    Function to read in and parse all ProcLog files associated with a given 
    process ID.  The contents of these files are returned as a collection of
    dictionaries ordered by:
      block name
        ProcLog name
           entry name
    """


    # Make sure we have a directory to load from
    baseDir = os.path.join('/dev/shm/bifrost/', str(pid))
    if not os.path.isdir(baseDir):
        raise RuntimeError("Cannot find log directory associated with PID %s" % pid)

    # Load
    contents = {}
    for parent,subnames,filenames in os.walk(baseDir):
        for filename in filenames:
            filename = os.path.join(parent, filename)

            ## Extract the block and logfile names
            logName = os.path.basename(filename)
            blockName = os.path.basename( os.path.dirname(filename) )
            if blockName == 'rings' and not include_rings:
                continue

            ## Load the file's contents
            try:
                subContents = load_by_filename(filename)
            except IOError:
                continue

            ## Save
            try:
                contents[blockName][logName] = subContents
            except KeyError:
                contents[blockName] = {logName:subContents}

    # Done
    return contents

def _get_command_line(pid):
    """
    Given a PID, use the /proc interface to get the full command line for 
    the process.  Return an empty string if the PID doesn't have an entry in
    /proc.
    """

    cmd = ''

    try:
        with open('/proc/%i/cmdline' % pid, 'r') as fh:
            cmd = fh.read()
            cmd = cmd.replace('\0', ' ')
            fh.close()
    except IOError:
        pass
    return cmd

class BifrostPipelines(object):
    """
    Class for monitoring the network receive and transmit rates for the 
    bifrost pipelines running on a machine.
    """
    
    def pipeline_pids(self):
        pidDirs = glob.glob(os.path.join(BIFROST_STATS_BASE_DIR, '*'))
        pidDirs.sort()
        
        pids = []
        for pidDir in pidDirs:
            pids.append( int(os.path.basename(pidDir), 10) )
        return pids
        
    def pipeline_count(self):
        return len(self.pipeline_pids())
        
    def pipelines(self):
        return [BifrostPipeline(pid) for pid in self.pipeline_pids()]

class BifrostPipeline(object):
    """
    Class for monitoring the network receive and transmit rates for the 
    bifrost pipeline with the provided process ID number.
    """
    
    def __init__(self, pid):
        self.pid = pid
        self.command = _get_command_line(self.pid)
        
    def _has_block(self, block):
        curr = self._get_state()
        return True if block in curr else False
        
    def _get_state(self):
        if not hasattr(self, '_state'):
            self._update_state()
            self._update_state()
            
        return self._state
        
    def _get_last_state(self):
        if not hasattr(self, '_state'):
            self._update_state()
            self._update_state()
            
        return self._last_state
        
    def _update_state(self):
        if not hasattr(self, '_state'):
            self._state = {}
        self._last_state = copy.deepcopy(self._state)
        
        contents = load_by_pid(self.pid)
        
        for block in contents.keys():
            if block[:3] != 'udp':
                continue
                
            t = time.time()
            try:
                log     = contents[block]['stats']
                good    = log['ngood_bytes']
                missing = log['nmissing_bytes']
                invalid = log['ninvalid_bytes']
                late    = log['nlate_bytes']
                nvalid  = log['nvalid']
            except KeyError:
                good, missing, invalid, late, nvalid = 0, 0, 0, 0, 0
                
            try:
                self._state[block]
            except KeyError:
                self._state[block] = {}
            self._state[block] = {'time'   : t, 
                                  'good'   : good, 
                                  'missing': missing, 
                                  'invalid': invalid, 
                                  'late'   : late, 
                                  'nvalid' : nvalid}

    def _get_rate(self, block, metric):
        # Make sure we have the block we are asked to report on 
        if not self._has_block(block):
            return 0
            
        # Make sure the data are current and enough time has passed that we 
        # know what is going on
        tNow = time.time()
        prev, curr = self._get_last_state(), self._get_state()
        if tNow - prev[block]['time'] > 30.0:
            self._update_state()
            prev, curr = self._get_last_state(), self._get_state()
        if curr[block]['time'] - prev[block]['time'] < 10.0:
            time.sleep(10)
            self._update_state()
            prev, curr = self._get_last_state(), self._get_state()
            print "->", tNow, curr[block]['time'],  prev[block]['time']
            
        # Compute the rate
        t0, metric0 = prev[block]['time'], prev[block][metric]
        t1, metric1 = curr[block]['time'], curr[block][metric]
        return (metric1-metric0)/(t1-t0)
        
    def rx_rate(self):
        """
        Get the current receive rate in B/s.
        """
        
        return self._get_rate('udp_capture', 'good')
    
    def tx_rate(self):
        """
        Get the current transmit rate in B/s.
        """
        
        return self._get_rate('udp_transmit', 'good')
        
    def _get_loss(self, block, snapshot=True):
        # Make sure we have the block we are asked to report on 
        if not self._has_block(block):
            return 0
            
        # Make sure the data are current and enough time has passed that we 
        # know what is going on
        tNow = time.time()
        prev, curr = self._get_last_state(), self._get_state()
        if tNow - prev[block]['time'] > 30.0:
            self._update_state()
            prev, curr = self._get_last_state(), self._get_state()
        if curr[block]['time'] - prev[block]['time'] < 10.0:
            time.sleep(10)
            self._update_state()
            prev, curr = self._get_last_state(), self._get_state()
            
        # Compute the loss
        if snapshot:
            good0, missing0 = prev[block]['good'], prev[block]['missing']
        else:
            good0, missing0 = 0, 0
        good1, missing1 = curr[block]['good'], curr[block]['missing']
        try:
            loss = (missing1-missing0)/float(good1-good0 + missing1-missing0)
        except ZeroDivisionError:
            loss = 0.0
        return max([0.0, min([loss, 1.0])])
        
    def rx_loss(self, snapshot=True):
        """
        Get the fractional receive loss.  If snapshot is False then
        the loss is integrated over the lifetime of the pipeline.
        """
        
        return self._get_loss('udp_capture', snapshot=snapshot)
        
if __name__ == "__main__":
    pipes = BifrostPipelines()
    for pipe in pipes.pipelines():
        print pipe, pipe.command, pipe.rx_rate(), pipe.rx_loss(), pipe.tx_rate()
        