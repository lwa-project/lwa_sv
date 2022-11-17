#!/usr/bin/env python3

import time
import socket
import struct

def sendSubsystemCommand(ss, cmd="RPT", data="SUMMARY"):
    """
    Use MCS/sch to send the given command to the specified subsystem.  For 
    'RPT' commands the value for the MIB is returned, otherwise the refernce
    ID is returned.
    """
    
    LWA_IP_MSE = '172.16.1.103'		# IP address of MCS Scheduler "ms_exec" process
    LWA_PORT_MSE = 9734				# Port for MCS Scheduler "ms_exec" process
    
    cmdStruct = struct.Struct('lliilliii256si')
    #  1 sid: subsystem ID
    #  2 ref: reference number
    #  3 cid: command ID
    #  4 scheduled? 1 = "do as close as possible to time in tv
    #  5 tv.tv_sec: epoch seconds
    #  6 tv.tv_usec: fractional remainder in microseconds
    #  7 response: see LWA_MSELOG_TP_*
    #  8 eSummary: see LWA_SIDSUM_*
    #  9 eMIBerror: > 0 on error; see LWA_MIBERR_*
    # 10 DATA on way out, R-COMMENT on way back
    # 11 datalen: -1 for (printable) string; 0 for zero-len;
    #    otherwise number of significant bytes
    
    # Convert the subsystem name to an MCS ID code
    if ss == 'ASP':
        sid = 12
    else:
        raise ValueError("Unknown subsystem ID: %s" % ss)
        
    # Convert the command name to a MCS ID code
    if cmd == "FPW":
        cid = 12
    else:
        raise ValueError("Unknown command: %s" % cmd)
        
    # Send the command
    try:    
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((LWA_IP_MSE, LWA_PORT_MSE))
        sock.settimeout(5)
        
        t = time.time()
        tv_sec, tv_usec = int(t), int((t % 1) * 1e6)
        mcscmd = cmdStruct.pack(sid, 0, cid, 0, tv_sec, tv_usec, 0, 0, 0, data.encode(), -1)
        sock.sendall(mcscmd)
        response = sock.recv(cmdStruct.size)
        response = cmdStruct.unpack(response)
        
        sock.close()
    except Exception as e:
        raise RuntimeError("MCS/exec - ms_exec does not appear to be running")
        
    # Wait a bit...
    time.sleep(0.2)
    
    return True


sendSubsystemCommand("ASP", "FPW", "000111")
time.sleep(1)
sendSubsystemCommand("ASP", "FPW", "000211")
