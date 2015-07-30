
import MCS2
import time
from Queue import Queue

def send_rpt(data):
	msg = MCS2.Msg(dst='ADP', cmd='RPT', data=data)
	print msg
	sender.put(msg)
	reply = receiver.get(timeout=0.5)
	print reply

if __name__ == "__main__":
	sender   = MCS2.MsgSender(("localhost",1742), subsystem='MCS')
	sender.input_queue = Queue()
	receiver = MCS2.MsgReceiver(("0.0.0.0",1743))
	sender.daemon = True
	receiver.daemon = True
	sender.start()
	receiver.start()
	
	send_rpt('SUMMARY')
	send_rpt('LASTLOG')
	send_rpt('VERSION')
	send_rpt('NOT_A_MIB_ENTRY')
	send_rpt('SERVER1_TEMP_AVG')
	#send_rpt('
	sender.request_stop()
	receiver.request_stop()
	sender.join()
	receiver.join()
