
import threading
import socket
import Queue

class UDPRecvThread(threading.Thread):
	STOP = '__UDPRecvThread_STOP__'
	def __init__(self, address, bufsize=16384):
		threading.Thread.__init__(self)
		self._addr      = address
		self._bufsize   = bufsize
		self._msg_queue = Queue.Queue() # For default behaviour
		self.socket     = socket.socket(socket.AF_INET,
		                                socket.SOCK_DGRAM)
		self.socket.bind(address)
	def request_stop(self):
		sendsock = socket.socket(socket.AF_INET,
		                         socket.SOCK_DGRAM)
		sendsock.connect(self._addr)
		sendsock.send(UDPRecvThread.STOP)
	def run(self):
		while True:
			pkt = self.socket.recv(self._bufsize)
			if pkt == UDPRecvThread.STOP:
				break
			self.process(pkt)
		self.shutdown()
	def process(self, pkt):
		"""Overide this in subclass"""
		self._msg_queue.put(pkt) # Default behaviour
	def shutdown(self):
		"""Overide this in subclass"""
		pass
	def get(self, timeout=None):
		try:
			return self._msg_queue.get(True, timeout)
		except Queue.Empty:
			return None

if __name__ == '__main__':
	port = 8321
	rcv = UDPRecvThread(("localhost", port))
	#rcv.daemon = True
	rcv.start()
	print "Waiting for packet on port", port
	pkt = rcv.get(timeout=5.)
	if pkt is not None:
		print "Received packet:", pkt
	else:
		print "Timed out waiting for packet"
	rcv.request_stop()
	rcv.join()
