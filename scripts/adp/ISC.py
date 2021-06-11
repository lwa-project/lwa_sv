# -*- coding: utf-8 -*-

"""
Inter-Server Communication (ISC) for ADP.  This allows messages to be send to 
the different servers for on-the-fly pipeline reconfiguration.
"""

from __future__ import print_function, absolute_import

import zmq
import time
import numpy
import binascii
import threading
from uuid import uuid4
from collections import deque


__version__ = '0.3'
__revision__ = '$Rev$'
__all__ = ['PipelineMessageServer', 'StartTimeClient', 'TriggerClient', 'TBNConfigurationClient',
           'DRXConfigurationClient', 'BAMConfigurationClient', 'CORConfigurationClient', 
           'PipelineSynchronizationServer',  'PipelineSynchronizationClient', 
           'PipelineEventServer', 'PipelineEventClient', '__version__', '__revision__', '__all__']


import sys
import logging
import functools
import traceback
try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO


from .AdpCommon import DATE_FORMAT, FS


def logException(func):
    logger = logging.getLogger('__main__')
    
    @functools.wraps(func)
    def tryExceptWrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
            
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            try:
                logger.error("%s in %s failed with %s at line %i", func, func.func_code.co_filename, str(e), func.func_code.co_firstlineno + 1)
            except AttributeError:
                logger.error("%s in %s failed with %s at line %i", func, func.__code__.co_filename, str(e), func.__code__.co_firstlineno + 1)
            
            # Grab the full traceback and save it to a string via StringIO
            fileObject = StringIO()
            traceback.print_tb(exc_traceback, file=fileObject)
            tbString = fileObject.getvalue()
            fileObject.close()
            
            # Print the traceback to the logger as a series of DEBUG messages
            for line in tbString.split('\n'):
                logger.debug("%s", line)
                
    return tryExceptWrapper


class PipelineMessageServer(object):
    """
    Class for broadcasting configuration information to the different pipelines 
    running on various servers.  This is implemented using 0MQ in a PUB/SUB
    scheme.
    """
    
    def __init__(self, addr=('adp', 5832), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind('tcp://*:%i' % addr[1])
        
    def packetStartTime(self, utcStartTime):
        """
        Send the UTC start time to TBN and DRX clients.
        """
        
        try:
            utcStartTime = utcStartTime.strftime(DATE_FORMAT)
        except AttributeError:
            pass
        self.socket.send('UTC %s' % utcStartTime)
        
    def tbnConfig(self, frequency, filter, gain):
        """
        Send TBN configuration information out to the clients.  This 
        includes:
          * frequency in Hz
          * filter code
          * gain setting
        """
        
        self.socket.send('TBN %.6f %i %i' % (frequency, filter, gain))
        
    def drxConfig(self, tuning, frequency, filter, gain):
        """
        Send DRX configuration information out to the clients.  This 
        includes:
          * the tuning number this update applied to
          * frequency in Hz
          * filter code
          * gain setting
        """
        
        self.socket.send('DRX %i %.6f %i %i' % (tuning, frequency, filter, gain))
        
    def bamConfig(self, beam, delays, gains, tuning, subslot):
        """
        Send BAM configuration information out to the clients.  This includes:
          * the beam number this update applies to
          * the delays as a 1-D numpy array
          * the gains as a 3-D numpy array
          * the tuning number this update applied to
          * the subslot in which the configuration is implemented
        """
        
        bDelays = binascii.hexlify( delays.tostring() )
        bGains = binascii.hexlify( gains.tostring() )
        self.socket.send('BAM %i %s %s %i %i' % (beam, bDelays, bGains, tuning, subslot))
        
    def corConfig(self, navg, tuning, gain, subslot):
        """
        Send COR configuration information out the clients.  This includes:
          * the integration time in units of subslots
          * the tuning number the update applies to
          * the gain
          * the subslot in which the configuration is implemented
        """
        
        self.socket.send('COR %i %i %i %i' % (navg, tuning, gain, subslot))
        
    def trigger(self, trigger, samples, mask, local=False):
        """
        Send a trigger to start dumping TBF data.  This includes:
          * the trigger time
          * the number of samples to dump
          * the DRX tuning mask to use
          * whether or not to dump to disk
        """
        
        self.socket.send('TRIGGER %i %i %i %i' % (trigger, samples, mask, local))
        
    def close(self):
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class PipelineMessageClient(object):
    """
    Client side version of PipelineMessageServer that is used to collect the 
    configuration updates as they come in.
    """
    
    @logException
    def __init__(self, group, addr=('adp', 5832), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.SUB)
        try:
            self.socket.setsockopt(zmq.SUBSCRIBE, group)
        except TypeError:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, group)
        self.socket.connect('tcp://%s:%i' % addr)
        
    @logException
    def __call__(self, block=False):
        """
        Pull in information if it is available.  If a message from the server
        is available it is returned.  Otherwise the behavior is determined by
        the 'block' keyword.  If 'block' is True, the function blocks until a 
        message is received.  If 'block' is False, False is returned 
        immediately.
        """
        
        try:
            msg = self.socket.recv(flags=(0 if block else zmq.NOBLOCK))
            return msg
        except zmq.error.ZMQError:
            return False
            
    @logException
    def close(self):
        """
        Close out the client.
        """
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class StartTimeClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for receiving 
    start time information."""
    
    def __init__(self, addr=('adp', 5832), context=None):
        super(StartTimeClient, self).__init__('UTC', addr=addr, context=context)
        
    def __call__(self):
        msg = super(StartTimeClient, self).__call__(block=True)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields = msg.split(None, 1)
            start = datetime.datetime.strptime(fields[1], DATE_FORMAT)
            return start


class TriggerClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for trigger 
    information.
    """
    
    def __init__(self, addr=('adp', 5832), context=None):
        super(TriggerClient, self).__init__('TRIGGER', addr=addr, context=context)
        
    def __call__(self, block=False):
        msg = super(TriggerClient, self).__call__(block=block)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields  = msg.split(None, 4)
            trigger = int(fields[1], 10)
            samples = int(fields[2], 10)
            mask    = int(fields[3], 10)
            local   = bool(int(fields[4], 10))
            return trigger, samples, mask, local


class TBNConfigurationClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for TBN 
    configuration information.
    """
    
    def __init__(self, addr=('adp', 5832), context=None):
        super(TBNConfigurationClient, self).__init__('TBN', addr=addr, context=context)
        
    def __call__(self):
        msg = super(TBNConfigurationClient, self).__call__(block=False)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields    = msg.split(None, 3)
            frequency = float(fields[1])
            filter    = int(fields[2], 10)
            gain      = int(fields[3], 10)
            return frequency, filter, gain


class DRXConfigurationClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for DRX 
    configuration information.
    """
    
    def __init__(self, addr=('adp', 5832), context=None):
        super(DRXConfigurationClient, self).__init__('DRX', addr=addr, context=context)
        
    def __call__(self):
        msg = super(DRXConfigurationClient, self).__call__(block=False)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields    = msg.split(None, 4)
            tuning    = int(fields[1], 10)
            frequency = float(fields[2])
            filter    = int(fields[3], 10)
            gain      = int(fields[4], 10)
            return tuning, frequency, filter, gain


class BAMConfigurationClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for BAM 
    configuration information.
    """
    
    def __init__(self, addr=('adp', 5832), context=None):
        super(BAMConfigurationClient, self).__init__('BAM', addr=addr, context=context)
        
    def __call__(self):
        msg = super(BAMConfigurationClient, self).__call__(block=False)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields = msg.split(None, 5)
            beam = int(fields[1], 10)
            delays = numpy.fromstring( binascii.unhexlify(fields[2]), dtype='>H' )
            delays.shape = (512,)
            gains = numpy.fromstring( binascii.unhexlify(fields[3]), dtype='>H' )
            gains.shape = (256,2,2)
            tuning = int(fields[4], 10)
            subslot = int(fields[5], 10)
            return beam, delays, gains, tuning, subslot


class CORConfigurationClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for COR 
    configuration information.
    """
    
    def __init__(self, addr=('adp', 5832), context=None):
        super(CORConfigurationClient, self).__init__('COR', addr=addr, context=context)
        
    def __call__(self):
        msg = super(CORConfigurationClient, self).__call__(block=False)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields  = msg.split(None, 3)
            navg    = int(fields[0], 10)
            tuning  = int(fields[1], 10)
            gain    = int(fields[2], 10)
            subslot = int(fields[3], 10)
            return navg, tuning, gain, subslot


class PipelineSynchronizationServer(object):
    """
    Class to provide packet-level synchronization across the pipelines.  
    This uses 0MQ in a ROUTER/DEALER setup to make sure clients reach the
    same point at the same time.
    """
    
    def __init__(self, nClients=6, addr=('adp', 5833), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind('tcp://*:%i' % addr[1])
        
        # Setup the client count
        self.nClients = nClients
        
        # Setup the threading
        self.thread = None
        self.alive = threading.Event()
        
    def start(self):
        """
        Start the synchronization pool.
        """
        
        if self.thread is not None:
            self.stop()
            
        self.thread = threading.Thread(target=self._sync, name='synchronizer')
        self.thread.setDaemon(1)
        self.alive.set()
        self.thread.start()
        
    def stop(self):
        """
        Stop the synchronization pool.
        """
        
        if self.thread is not None:
            self.alive.clear()
            self.thread.join()
            
            self.thread = None
            
    def _sync(self):
        clients = []
        nAct = 0
        nRecv = 0
        
        while self.alive.isSet():
            client, msg = self.socket.recv_multipart()
            if msg == 'JOIN':
                if client not in clients:
                    clients.append( client )
                    nAct += 1
                    print("FOUND '%s'" % client)
                    
            elif msg == 'LEAVE':
                try:
                    del clients[clients.index(client)]
                    nAct -= 1
                    print("LOST '%s'" % client)
                except ValueError:
                    pass
                    
            elif msg[:3] == 'TAG':
                nRecv += 1
                
                if nRecv == nAct:
                    for client in clients:
                        self.socket.send_multipart([client, msg])
                        
    def close(self):
        """
        Stop the synchronization pool and close out the server.
        """
        
        self.stop()
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class PipelineSynchronizationClient(object):
    """
    Client class for PipelineSynchronizationClient.
    """
    
    @logException
    def __init__(self, id=None, addr=('adp', 5833), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.DEALER)
        if id is not None:
            try:
                self.socket.setsockopt(zmq.IDENTITY, str(id))
            except TypeError:
                self.socket.setsockopt_string(zmq.IDENTITY, str(id))
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect('tcp://%s:%i' % addr)
        
        # Connect to the server
        self.socket.send('JOIN')
        
    @logException
    def __call__(self, tag=None):
        self.socket.send('TAG:%s' % tag)
        return self.socket.recv()
        
    @logException
    def close(self):
        """
        Leave the synchronization pool and close out the client.
        """
        
        self.socket.send('LEAVE')
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class PipelineEventServer(object):
    """
    Class to provide a distributed event across the pipelines.  This uses 
    0MQ in a REQUEST/REPLY setup to make sure clients can lock/unlock each
    other to control data flow.
    """
    
    def __init__(self, addr=('adp', 5834), context=None, timeout=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind('tcp://*:%i' % addr[1])
        
        # Setup the poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        
        # Setup the event
        self.timeout = timeout
        self._state = {}
        
        # Setup the threading
        self.thread = None
        self.alive = threading.Event()
        
    def _set(self, id):
        self._state[id] = time.time()
        return True
        
    def _is_set(self, id):
        if len(self._state):
            if self.timeout is None:
                return True
                
            else:
                for id in sorted(self._state):
                    age = time.time() - self._state[id]
                    if age <= self.timeout:
                        return True
                    else:
                        self._clear(id)
                        
        return False
        
    def _clear(self, id):
        try:
            del self._state[id]
            return True
            
        except KeyError:
            return False
            
    @logException
    def start(self):
        """
        Start the event pool.
        """
        
        if self.thread is not None:
            self.stop()
            
        self._state = {}
        
        self.thread = threading.Thread(target=self._event, name='event')
        self.thread.setDaemon(1)
        self.alive.set()
        self.thread.start()
        
    @logException
    def stop(self):
        """
        Stop the event pool.
        """
        
        if self.thread is not None:
            self.alive.clear()
            self.thread.join()
            
            self.thread = None
            
    @logException
    def _event(self):
        while self.alive.isSet():
            msg = dict(self.poller.poll(1000))
            if msg:
                if msg.get(self.socket) == zmq.POLLIN:
                    msg = self.socket.recv(zmq.NOBLOCK)
                    id, msg = msg.split(None, 1)
                    
                    if msg == 'SET':
                        status = self._set(id)
                    elif msg == 'CLEAR':
                        status = self._clear(id)
                    elif msg == 'IS_SET':
                        status = self._is_set(id)
                    elif msg == 'LEAVE':
                        status = self._clear(id)
                    else:
                        status = False
                    try:
                        self.socket.send(str(status))
                    except TypeError:
                        self.socket.send_string(str(status))
                    
    def close(self):
        """
        Stop the locking pool and close out the server.
        """
        
        self.stop()
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class PipelineEventClient(object):
    """
    Client class for PipelineEventClient.
    """
    
    @logException
    def __init__(self, id=None, addr=('adp', 5834), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.REQ)
        if id is None:
            id = uuid4()
        try:
            self.socket.setsockopt(zmq.IDENTITY, str(id))
        except TypeError:
            self.socket.setsockopt_string(zmq.IDENTITY, str(id))
        self.socket.setsockopt(zmq.LINGER, 100)
        self.socket.connect('tcp://%s:%i' % addr)
        
        # Save the ID
        self.id = self.socket.getsockopt(zmq.IDENTITY)
        
    @logException
    def is_set(self):
        try:
            self.socket.send('%s %s' % (self.id, 'IS_SET'))
        except TypeError:
            self.socket.send_string('%s %s' % (self.id, 'IS_SET'))
        return True if self.socket.recv() == 'True' else False
        
    @logException
    def isSet(self):
        return self.is_set()
        
    @logException
    def set(self):
        try:
            self.socket.send('%s %s' % (self.id, 'SET'))
        except TypeError:
            self.socket.send_string('%s %s' % (self.id, 'SET'))
        return True if self.socket.recv() == 'True' else False
        
    @logException
    def clear(self):
        try:
            self.socket.send('%s %s' % (self.id, 'CLEAR'))
        except TypeError:
            self.socket.send_string('%s %s' % (self.id, 'CLEAR'))
        return True if self.socket.recv() == 'True' else False
        
    @logException
    def wait(self, timeout=None):
        t0 = time.time()
        while not self.is_set():
            time.sleep(0.01)
            t1 = time.time()
            if timeout is not None:
                if t1-t0 > timeout:
                    return False
        return True
        
    @logException
    def close(self):
        """
        Leave the synchronization pool and close out the client.
        """
        
        try:
            self.socket.send('%s %s' % (self.id, 'LEAVE'))
        except TypeError:
            self.socket.send_string('%s %s' % (self.id, 'LEAVE'))
        status = True if self.socket.recv() in ('True', b'True') else False
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()
            
        return status


class InternalTrigger(object):
    """
    Class for signaling that a potentially interesting event has occurred.  
    This is sent to the InternalTrigger server for validation.  This is 
    implemented using 0MQ in a PUSH/PULL scheme.
    """
    
    def __init__(self, id=None, addr=('adp', 5835), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.PUSH)
        if id is None:
            id = uuid4()
        try:
            self.socket.setsockopt(zmq.IDENTITY, str(id))
        except TypeError:
            self.socket.setsockopt_string(zmq.IDENTITY, str(id))
        self.socket.setsockopt(zmq.LINGER, 10)
        self.socket.connect('tcp://%s:%i' % addr)
        
        # Save the ID
        self.id = self.socket.getsockopt(zmq.IDENTITY)
        
    def __call__(self, timestamp):
        """
        Send the event's timestamp as a DP/ADP timestamp value, i.e., 
        int(UNIX time * 196e6).
        """
        
        try:
            self.socket.send('%s %s' % (self.id, str(timestamp)))
        except TypeError:
            self.socket.send_string('%s %s' % (self.id, str(timestamp)))
        
    def close(self):
        """
        Leave the triggering pool and close out the client.
        """
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class InternalTriggerProcessor(object):
    """
    Class to gather triggers from various InternalTrigger clients, validate 
    them, and actually act on the trigger.
    """
    
    def __init__(self, port=5835, coincidence_window=5e-4, min_coincident=6, deadtime=60.0, callback=None, context=None):
        # Set the port to use
        self.port = port
        
        # Set the coincidence window time limit (window size used to determine 
        # if the triggers occurred at the same time)
        self.coincidence_window = int(float(coincidence_window)*FS)
        
        # Set the minimum number of coincident events within the time window to 
        # accept as real events
        self.min_coincident = int(min_coincident)
        
        # Set the deadtime (required downtime between valid triggers) and 
        # callback (function to call on a valid trigger)
        self.deadtime = int(float(deadtime)*FS)
        self.callback = callback
        
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the event list
        self.events = {}
        
        # Setup the thread control
        self.shutdown_event = threading.Event()
        
    def shutdown(self):
        self.shutdown_event.set()
        
    def run(self):
        tLast = 0
        
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 10)
        self.socket.bind('tcp://*:%i' % self.port)
        
        # Setup the poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        
        while not self.shutdown_event.is_set():
            # Get an event and parse it out
            msg = dict(self.poller.poll(5000))
            if msg:
                if msg.get(self.socket) == zmq.POLLIN:
                    msg = self.socket.recv(zmq.NOBLOCK)
                    try:
                        id, timestamp = msg.split(None, 1)
                        timestamp = int(timestamp, 10)
                    except ValueError:
                        continue
                        
                    # Ignore events that occurring during the mandatory deadtime
                    if timestamp - tLast < self.deadtime:
                        continue
                        
                    # Store the event
                    self.events[id] = timestamp
                    
                    # Validate the event(s)
                    count = len(self.events)
                    newest = max(self.events.values())
                    oldest = min(self.events.values())
                    diff = newest - oldest
                    if count >= self.min_coincident and diff <= self.coincidence_window:
                        ## Looks like we have an event, update the state and send the 
                        ## trigger out
                        tLast = newest
                        self.events.clear()
                        if self.callback is not None:
                            self.callback(oldest)
                            
        # Close out the socket
        self.socket.close()
        if self.newContext:
            self.context.destroy()
