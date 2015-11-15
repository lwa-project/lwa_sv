
import Queue
import time
from datetime import datetime
import socket
from ConsumerThread import ConsumerThread
from SocketThread import UDPRecvThread
import string

# Maximum number of bytes to receive from MCS
MCS_RCV_BYTES = 16*1024

# Note: Unless otherwise noted, slots are referenced to Unix time
def get_current_slot():
	# Returns current slot in Unix time
	return int(time.time())
def get_current_mpm():
	# Returns current milliseconds past midnight as an integer
	dt = datetime.utcnow()
	ms = int(dt.microsecond / 1000.)
	return ((dt.hour*60 + dt.minute)*60 + dt.second)*1000 + ms
def slot2utc(slot=None):
	if slot is None:
		slot = get_current_slot()
	return time.gmtime(slot)
# TODO: What is 'station time'? Is it UTC?
def slot2dayslot(slot=None):
	utc     = slot2utc(slot)
	dayslot = (utc.tm_hour*60 + utc.tm_min)*60 + utc.tm_sec
	return dayslot
def slot2mpm(slot=None):
	return slot2dayslot(slot) * 1000
def slot2mjd(slot=None):
	tt = slot2utc(slot)
	# Source: SkyField
	janfeb = tt.tm_mon < 3
	jd_day = tt.tm_mday
	jd_day += 1461 *  (tt.tm_year + 4800 - janfeb) / 4
	jd_day +=  367 *  (tt.tm_mon  -    2 + janfeb * 12) / 12
	jd_day -=    3 * ((tt.tm_year + 4900 - janfeb) / 100) / 4
	jd_day -= 32075
	mjd = tt.tm_sec
	mjd = mjd*(1./60) + tt.tm_min
	mjd = mjd*(1./60) + tt.tm_hour
	mjd = mjd*(1./24) + (jd_day - 2400000.5)
	mjd -= 0.5
	return mjd

def mib_parse_label(data):
	"""Splits an MIB label into a list of arguments
	     E.g., "ANT71_TEMP_MAX" --> ['ANT', 71, 'TEMP', 'MAX']
	"""
	args = []
	arg = ''
	mode = None
	for c in data:
		if c in string.ascii_uppercase:
			if mode == 'i':
				args.append(int(arg))
				arg = ''
			arg += c
			mode = 's'
		elif c in string.digits:
			if mode == 's':
				args.append(arg)
				arg = ''
			arg += c
			mode = 'i'
		elif c == '_':
			if mode is not None:
				args.append(int(arg) if mode == 'i' else arg)
				arg = ''
			mode = None
	args.append(int(arg) if mode == 'i' else arg)
	key = mib_args2key(args)
	return key, args
def mib_args2key(args):
	"""Merges an MIB label arg list back into a label suitable for
	     use as a lookup key (all indexes are removed)."""
	return '_'.join([arg for arg in args if not isinstance(arg, int)])

class Msg(object):
	count = 0
	# Note: MsgSender will automatically set src
	def __init__(self, illegal_argument=None,
	             src=None, dst=None, cmd=None, ref=None, data='', dst_ip=None,
	             pkt=None, src_ip=None):
		assert(illegal_argument is None) # Ensure named args only
		self.dst  = dst
		self.src  = src
		self.cmd  = cmd
		self.ref  = ref
		if self.ref is None:
			self.ref = Msg.count % 10**9
			Msg.count += 1
		self.mjd  = None
		self.mpm  = None
		self.data = data
		self.dst_ip = dst_ip
		self.slot = None # For convenience, not part of encoded pkt
		if pkt is not None:
			self.decode(pkt)
		self.src_ip = src_ip
	def __str__(self):
		if self.slot is None:
			return ("<MCS Msg %i: '%s' from %s to %s, data='%s'>" %
			        (self.ref, self.cmd, self.src, self.dst,
			         self.data))
		else:
			return (("<MCS Msg %i: '%s' from %s (%s) to %s, data='%s', "+
			        "rcv'd in slot %i>") %
			        (self.ref, self.cmd, self.src, self.src_ip,
			         self.dst, self.data,
			         self.slot))
	def decode(self, pkt):
		self.slot = get_current_slot()
		self.dst  = pkt[:3]
		self.src  = pkt[3:6]
		self.cmd  = pkt[6:9]
		self.ref  = int(pkt[9:18])
		datalen   = int(pkt[18:22])
		self.mjd  = int(pkt[22:28])
		self.mpm  = int(pkt[28:37])
		space     = pkt[37]
		self.data = pkt[38:38+datalen]
		# WAR for DATALEN parameter being wrong for BAM commands (FST too?)
		broken_commands = ['BAM']#, 'FST']
		if self.cmd in broken_commands:
			self.data = pkt[38:]
	def create_reply(self, accept, status, data=''):
		msg = Msg(#src=self.dst,
		          dst=self.src,
		          cmd=self.cmd,
		          ref=self.ref,
		          dst_ip=self.src_ip)
		#msg.mjd, msg.mpm = getTime()
		response = 'A' if accept else 'R'
		msg.data = response + str(status).rjust(7) + data
		return msg
	def is_valid(self):
		return (self.dst is not None and len(self.dst) <= 3 and
		        self.src is not None and len(self.src) <= 3 and
		        self.cmd is not None and len(self.cmd) <= 3 and
		        self.ref is not None and (0 <= self.ref < 10**9) and
		        self.mjd is not None and (0 <= self.mjd < 10**6) and
		        self.mpm is not None and (0 <= self.mpm < 10**9) and
		        len(self.data) < 10**4)
	def encode(self):
		self.mjd = int(slot2mjd())
		self.mpm = get_current_mpm()
		assert( self.is_valid() )
		pkt = (self.dst.ljust(3) +
		       self.src.ljust(3) +
		       self.cmd.ljust(3) +
		       str(self.ref      ).rjust(9) +
		       str(len(self.data)).rjust(4) +
		       str(self.mjd      ).rjust(6) +
		       str(self.mpm      ).rjust(9) +
		       ' ' +
		       self.data)
		return pkt

class MsgReceiver(UDPRecvThread):
	def __init__(self, address, subsystem='ALL'):
		UDPRecvThread.__init__(self, address)
		self.subsystem = subsystem
		self.msg_queue = Queue.Queue()
		self.name      = 'MCS.MsgReceiver'
	def process(self, pkt, src_ip):
		if len(pkt):
			msg = Msg(pkt=pkt, src_ip=src_ip)
			if ( self.subsystem == 'ALL' or
			     msg.dst        == 'ALL' or
			     self.subsystem == msg.dst ):
				self.msg_queue.put(msg)
	def shutdown(self):
		self.msg_queue.put(ConsumerThread.STOP)
		print self.name, "shutdown"
	def get(self, timeout=None):
		try:
			return self.msg_queue.get(True, timeout)
		except Queue.Empty:
			return None

class MsgSender(ConsumerThread):
	def __init__(self, dst_addr, subsystem,
	             max_attempts=5):
		ConsumerThread.__init__(self)
		self.subsystem    = subsystem
		self.max_attempts = max_attempts
		self.socket       = socket.socket(socket.AF_INET,
		                                  socket.SOCK_DGRAM)
		#self.socket.connect(address)
		self.dst_ip   = dst_addr[0]
		self.dst_port = dst_addr[1]
		self.name = 'MCS.MsgSender'
	def process(self, msg):
		msg.src  = self.subsystem
		pkt      = msg.encode()
		dst_ip   = msg.dst_ip if msg.dst_ip is not None else self.dst_ip
		dst_addr = (dst_ip, self.dst_port)
		#print "Sending msg to", dst_addr
		for attempt in xrange(self.max_attempts-1):
			try:
				#self.socket.send(pkt)
				self.socket.sendto(pkt, dst_addr)
			except socket.error:
				time.sleep(0.001)
			else:
				return
		#self.socket.send(pkt)
		self.socket.sendto(pkt, dst_addr)
	def shutdown(self):
		print self.name, "shutdown"
