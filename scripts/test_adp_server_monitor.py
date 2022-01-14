#!/usr/bin/env python

from __future__ import print_function

import zmq

if __name__ == "__main__":
	
	zmqctx = zmq.Context()
	sock = zmqctx.socket(zmq.REQ)
	sock.connect('tcp://localhost:1752')
	sock.send('PNG')
	print(sock.recv_json())
	sock.send('TEMP_MIN')
	print(sock.recv_json())
	sock.send('TEMP_MAX')
	print(sock.recv_json())
	sock.send('TEMP_AVG')
	print(sock.recv_json())
	sock.send('STAT')
	print(sock.recv_json())
	sock.send('INFO')
	print(sock.recv_json())
	sock.send('SOFTWARE')
	print(sock.recv_json())
	
	
