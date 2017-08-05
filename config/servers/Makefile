
include ../common.mk

# TODO: See ledaovro:/root/at_reboot.sh and /root/set_irqs.sh
#         nvidia-smi; nvidia-smi -pm 1
#         /sbin/rmmod mlx4_core; /sbin/modprobe mlx4_en; /sbin/ifup eth2

all:

configure: configure_grub network resolvconf configure_hosts ssh nfs_client ntp sysctl socket_buffers irq_affinity
configure_partial: ntp sysctl socket_buffers irq_affinity rclocal
