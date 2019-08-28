
from socket import gethostbyname

def ip2int(ipaddr):
    """ Convert IP address to integer for ROACH programming """
    ipb = [int(ipbit) for ipbit in ipaddr.split('.')]
    assert len(ipb) == 4
    int_addr = ipb[0]*(1<<24) + ipb[1]*(1<<16) + ipb[2]*(1<<8) + ipb[3]
    return int_addr

def mac2int(macaddr):
    """ Convert IP address to integer for ROACH programming
    Address should be of form 00:01:02:03:04:05"""
    m = [int(mb, 16) for mb in macaddr.split(':')]
    assert len(m) == 6
    int_addr = (m[0]*(1<<40) + m[1]*(1<<32) + m[2]*(1<<24) +
                m[3]*(1<<16) + m[4]*(1<< 8) + m[5])
    return int_addr

def host2ip(host):
    return gethostbyname(host)

def load_ethers(filename="/etc/ethers"):
    macs = {}
    with open(filename, 'r') as f:
        for line in f:
            mac, host = line.split()
            ip = host2ip(host)
            macs[ip] = mac
    return macs

def gen_arp_table(ips, macs):
    """ Generates a 256-entry IP->MAC mapping table"""
    arp_table_ints = [mac2int('ff:ff:ff:ff:ff:ff') for i in range(256)]
    for ip, mac in zip(ips, macs):
        ip_idx = ip2int(ip) % 256
        arp_table_ints[ip_idx] = mac2int(mac)
    return arp_table_ints
