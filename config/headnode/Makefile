
include ../common.mk

# TODO: See ledaovro:/root/at_reboot.sh and /root/set_irqs.sh
#         nvidia-smi; nvidia-smi -pm 1
#         /sbin/rmmod mlx4_core; /sbin/modprobe mlx4_en; /sbin/ifup eth2

all:

configure: configure_grub iptables network resolvconf dnsmasq configure_hosts configure_ethers ssh nfs_server ntp dev_ttyUSB0 sysctl socket_buffers
