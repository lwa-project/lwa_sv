
# System configuration tasks for ADP headnode and servers

GRUB_CONF                 ?= /etc/default/grub
NETWORK_CONF              ?= /etc/network/interfaces
RESOLVCONF                ?= /etc/resolvconf/resolv.conf.d/tail
DNSMASQ_CONF              ?= /etc/dnsmasq.conf
HOSTS_CONF                ?= /etc/hosts
ETHERS_CONF               ?= /etc/ethers
IPTABLES_CONF             ?= /etc/iptables.rules
MGMT_SUBNET               ?= 169.254.128.0/24
SSHD_CONFIG               ?= /etc/ssh/sshd_config
SSH_CONFIG                ?= /etc/ssh/ssh_config
NTP_CONF                  ?= /etc/ntp.conf
LOCAL_NTP_SERVER          ?= 10.1.1.50 # Optional
SOCK_WMEM_CONF            ?= /proc/sys/net/core/wmem_max
SOCK_RMEM_CONF            ?= /proc/sys/net/core/rmem_max
SOCK_BUF_LIMIT            ?= "536870912"

all:

$(GRUB_CONF): ./grub
	cp $< $@
	update-grub
grub: $(GRUB_CONF)
.PHONY grub

$(IPTABLES_CONF):
	iptables -A FORWARD -s $(MGMT_SUBNET) -i eth1 -o eth0 -m conntrack --ctstate NEW -j ACCEPT
	iptables -A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
	iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
	sh -c "iptables-save > $@"
iptables: $(IPTABLES_CONF)
.PHONY iptables

$(NETWORK_CONF): ./interfaces
	cp $< $@
	ifdown --exclude=lo -a && ifup --exclude=lo -a # Restart networking
network: $(NETWORK_CONF) iptables
.PHONY network

$(RESOLVCONF): ./resolv.conf.d_tail
	cp $< $@
	resolvconf -u
resolvconf: $(RESOLVCONF)
.PHONY resolvconf

$(DNSMASQ_CONF): ./dnsmasq.conf network
	cp $< $@
	service dnsmasq restart
dnsmasq: $(DNSMASQ_CONF)
.PHONY dnsmasq

$(HOSTS_CONF): ../hosts network
	cp $< $@
hosts: $(HOSTS_CONF)
.PHONY: hosts

$(ETHERS_CONF): ./ethers network
	cp $< $@
ethers: $(ETHERS_CONF)
.PHONY: ethers

$(SSHD_CONF): ./sshd_config
	cp $< $@
$(SSH_CONF): ./ssh_config
	cp $< $@
ssh: $(SSHD_CONF) $(SSH_CONF) ../ssh_hosts network
	ssh-keyscan -t rsa,dsa,ecdsa -f ../ssh_hosts | sort -u - /etc/ssh/ssh_known_hosts | tee /etc/ssh/ssh_known_hosts
	restart ssh
.PHONY: ssh

nfs_server: network
	apt-get install -y nfs-kernel-server
	mkdir -p /export/home
	chmod 777 /export
	chmod 777 /export/home
	grep -q "/home"        /etc/fstab   || echo "/home           /export/home    none    bind            0       0" >> /etc/fstab
	grep -q "/export/home" /etc/exports || printf "# Note: no_root_squash is not a great idea, but is needed to allow clients to 'emacs blah'\n    /export/home 169.254.128.0/24(rw,nohide,insecure,no_subtree_check,async,no_root_squash)" >> /etc/exports
	service nfs-kernel-server restart
.PHONY: nfs_server

nfs_client: network
	apt-get install -y nfs-common
	grep -q "adp:/export/home" /etc/fstab || echo "adp:/export/home	/home			nfs	auto	0 0" >> /etc/fstab
	mount -a -t nfs
.PHONY: nfs_client

$(NTP_CONF): ./ntp.conf network
	/usr/sbin/ntpdate $(LOCAL_NTP_SERVER) ntp.ubuntu.com pool.ntp.org # Initial estimate
	apt-get install -y ntp
	cp $< $@
	restart ntp
	ntpq -p
ntp: $(NTP_CONF_FILE)
.PHONY: ntp

dev_ttyUSB0:
	usermod -a -G dialout adp # Allow access to /dev/ttyUSB0
.PHONY: dev_ttyUSB0

# Note that setsockopt(SO_SND/RCVBUF) actually allocates double the requested amount
$(SOCK_WMEM_CONF):
	echo $(SOCK_BUF_LIMIT) | sudo tee $@
$(SOCK_RMEM_CONF):
	echo $(SOCK_BUF_LIMIT) | sudo tee $@
socket_buffers: $(SOCK_WMEM_CONF) $(SOCK_RMEM_CONF)
.PHONY: socket_buffers
