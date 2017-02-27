#!/usr/bin/env python

import MCS2
import time
from Queue import Queue
import struct

MSG_REPLY_TIMEOUT = 5

def send_rpt(data):
	msg = MCS2.Msg(dst='ADP', cmd='RPT', data=data)
	print msg
	sender.put(msg)
	reply = receiver.get(timeout=MSG_REPLY_TIMEOUT)
	print reply
	#print reply.data, len(reply.data)
	if reply is not None and len(reply.data)-8 == 4:
		print struct.unpack('>f', reply.data[8:])

def send_msg(cmd, data=''):
        msg = MCS2.Msg(dst='ADP', cmd=cmd, data=data)
        print msg
        sender.put(msg)
        reply = receiver.get(timeout=MSG_REPLY_TIMEOUT)
	print reply

if __name__ == "__main__":
	import sys
	sender   = MCS2.MsgSender(("localhost",1742), subsystem='MCS')
	sender.input_queue = Queue()
	receiver = MCS2.MsgReceiver(("0.0.0.0",1743))
	sender.daemon = True
	receiver.daemon = True
	sender.start()
	receiver.start()
	
	if len(sys.argv) > 1 and sys.argv[1] == 'status':
		# Continuously query status
		while True:
			send_rpt('SUMMARY')
			send_rpt('INFO')
			send_rpt('LASTLOG')
			time.sleep(3)
	elif len(sys.argv) > 2:
		if sys.argv[1] == 'RPT':
			send_rpt(sys.argv[2])
		elif sys.argv[1] == 'TBN':
			freq, bw, gain, subslot = sys.argv[2].split(None, 3)
			freq = float(freq)
			bw = int(bw, 10)
			gain = int(gain, 10)
			subslot = int(subslot, 10)
			data = struct.pack('>fhh', freq, bw, gain)
			send_msg(sys.argv[1], data)
		elif sys.argv[1] == 'DRX':
			tune, freq, bw, gain, subslot = sys.argv[2].split(None, 4)
			tune = int(tune, 10)
			freq = float(freq)
			bw = int(bw, 10)
			gain = int(gain, 10)
			subslot = int(subslot, 10)
			data = struct.pack('>BfBh', tune, freq, bw, gain)
			send_msg(sys.argv[1], data)
		elif sys.argv[1] == 'TBF':
			bits, tdump, nsamp, mask = sys.argv[2].split(None, 3)
			bits = int(bits, 10)
			tdump = int(tdump, 10)
			nsamp = int(nsamp, 10)
			mask = int(mask, 10)
			data = struct.pack('>Biiq', bits, tdump, nsamp, mask)
			send_msg(sys.argv[1], data)
		elif sys.argv[1] == 'INI':
			data = ' '.join(sys.argv[2:])
			send_msg(sys.argv[1], data)
		sys.exit()	
	elif len(sys.argv) > 1:
		# Send specific command with optional data
		if len(sys.argv) > 2:
			data = ' '.join(sys.argv[2:])
			send_msg(sys.argv[1], data)
		else:
			send_msg(sys.argv[1])
		for i in xrange(5):
			send_rpt('SUMMARY')
			send_rpt('INFO')
			send_rpt('LASTLOG')
			time.sleep(1)
		sys.exit(0)
	
	send_rpt('SUMMARY')
	send_rpt('INFO')
	send_rpt('LASTLOG')
	send_rpt('VERSION')
	send_rpt('BOARD1_FIRMWARE')
	send_rpt('NOT_A_MIB_ENTRY')
	send_rpt('SERVER1_TEMP_AVG')
	send_rpt('ANT1_RMS')
	send_rpt('ANT2_RMS')
	send_rpt('ANT32_RMS')
	send_rpt('ANT33_RMS')
	send_rpt('ANT34_RMS')
	send_rpt('ANT1_SAT')
	send_rpt('ANT2_SAT')
	send_rpt('ANT1_PEAK')
	send_rpt('ANT2_PEAK')
	send_rpt('ANT1_DCOFFSET')
	send_rpt('ANT2_DCOFFSET')
	send_rpt('BOARD1_TEMP_AVG')
	send_rpt('BOARD1_TEMP_MIN')
	send_rpt('BOARD1_TEMP_MAX')
        send_rpt('BOARD2_TEMP_AVG')
	send_rpt('BOARD2_TEMP_MIN')
	send_rpt('BOARD2_TEMP_MAX')
	send_rpt('GLOBAL_TEMP_MAX')
	send_rpt('GLOBAL_TEMP_MIN')
	send_rpt('GLOBAL_TEMP_AVG')
 	
	#send_rpt('
	sender.request_stop()
	receiver.request_stop()
	sender.join()
	receiver.join()
