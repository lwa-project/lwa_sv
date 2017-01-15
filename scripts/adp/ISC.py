# -*- coding: utf-8 -*-

"""
Inter-Server Communication (ISC) for ADP.  This allows messages to be send to 
the different servers for on-the-fly pipeline reconfiguration.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import zmq
import time
import numpy
import binascii
import threading


__version__ = '0.1'
__revision__ = '$Rev$'
__all__ = ['PipelineMessageServer', 'StartTimeClient', 'TriggerClient', 'TBNConfigurationClient',
		 'DRXConfigurationClient', 'BAMConfigurationClient', 'PipelineSynchronizationServer',
		 'PipelineSynchronizationClient', 
		 '__version__', '__revision__', '__all__']


import sys
import logging
import functools
import traceback
try:
	import cStringIO as StringIO
except ImportError:
	import StringIO


from AdpCommon import DATE_FORMAT


def logException(func):
	logger = logging.getLogger('__main__')
	
	@functools.wraps(func)
	def tryExceptWrapper(*args, **kwargs):
		try:
			return func(*args, **kwargs)
			
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			logger.error("%s in %s failed with %s at line %i", func, func.func_code.co_filename, str(e), func.func_code.co_firstlineno + 1)
			
			# Grab the full traceback and save it to a string via StringIO
			fileObject = StringIO.StringIO()
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
		
		self.socket.send('TBN %.3f %i %i' % (frequency, filter, gain))
		
	def drxConfig(self, tuning, frequency, filter, gain):
		"""
		Send DRX configuration information out to the clients.  This 
		includes:
		  * the tuning number this update applied to
		  * frequency in Hz
		  * filter code
		  * gain setting
		"""
		
		self.socket.send('DRX %i %.3f %i %i' % (tuning, frequency, filter, gain))
		
	def bamConfig(self, beam, delays, gains, tuning):
		"""
		Send BAM configuration information out to the clients.  This includes:
		  * the beam number this update applies to
		  * the delays as a 1-D numpy array
		  * the gains as a 3-D numpy array
		  * the tuning number this update applied to
		"""
		
		bDelays = binascii.hexlify( delays.tostring() )
		bGains = binascii.hexlify( gains.tostring() )
		self.socket.send('BAM %i %s %s %i' % (beam, bDelays, bGains, tuning))
		
	def trigger(self, samples):
		"""
		Send a trigger to start dumping TBF data.  This includes:
		  * the number of samples to dump
		"""
		
		self.socket.send('TRIGGER %s %i' % (time.time(), samples))
		
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
		self.socket.setsockopt(zmq.SUBSCRIBE, group)
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
		
	def __call__(self):
		msg = super(TriggerClient, self).__call__(block=True)
		if not msg:
			# Nothing to report
			return False
		else:
			# Unpack
			fields = msg.split(None, 2)
			tTrigger = float(fields[1])
			samples = int(fields[2], 10)
			return tTrigger, samples


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
			fields = msg.split(None, 3)
			frequency = float(fields[1])
			filter = int(fields[2], 10)
			gain = int(fields[3], 10)
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
			fields = msg.split(None, 4)
			tuning = int(fields[1], 10)
			frequency = float(fields[2])
			filter = int(fields[3], 10)
			gain = int(fields[4], 10)
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
			fields = msg.split(None, 4)
			beam = int(fields[1], 10)
			delays = numpy.fromstring( binascii.unhexlify(fields[2]), dtype='>H' )
			delays.shape = (512,)
			gains = numpy.fromstring( binascii.unhexlify(fields[3]), dtype='>H' )
			gains.shape = (256,2,2)
			tuning = int(fields[4], 10)
			return beam, delays, gains, tuning


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
		
		while self.alive.isSet():
			client, msg = self.socket.recv_multipart()
			if msg == 'JOIN':
				if client not in clients:
					clients.append( client )
					print "FOUND '%s'" % client
					
			elif msg == 'LEAVE':
				try:
					del clients[clients.index(client)]
					print "LOST '%s'" % client
				except ValueError:
					pass
					
			elif msg[:3] == 'TAG':
				if len(clients) == self.nClients:
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
			self.socket.setsockopt(zmq.IDENTITY, str(id))
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
