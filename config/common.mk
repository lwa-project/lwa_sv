
# System configuration tasks for ADP headnode and servers

GRUB_CONF                 ?= /etc/default/grub
NETWORK_CONF              ?= /etc/network/interfaces
RESOLVCONF                ?= /etc/resolvconf/resolv.conf.d/tail
DNSMASQ_CONF              ?= /etc/dnsmasq.conf
HOSTS_CONF                ?= /etc/hosts
ETHERS_CONF               ?= /etc/ethers
IPTABLES_CONF             ?= /etc/iptables.rules
RCLOCAL_CONF              ?= /etc/rc.local
MGMT_SUBNET               ?= 169.254.128.0/24
SSHD_CONFIG               ?= /etc/ssh/sshd_config
SSH_CONFIG                ?= /etc/ssh/ssh_config
NTP_CONF                  ?= /etc/ntp.conf
LOCAL_NTP_SERVER          ?= 10.1.1.50 # Optional
SOCK_WMEM_CONF            ?= /proc/sys/net/core/wmem_max
SOCK_RMEM_CONF            ?= /proc/sys/net/core/rmem_max
SOCK_BUF_LIMIT            ?= "536870912"
SYSCTL_CONF               ?= /etc/sysctl.conf
DATA_NETWORK_IFACE        ?= p5p1
IRQBALANCE_CONF           ?= /etc/default/irqbalance

all:

.PHONY: configure_grub
$(GRUB_CONF): ./grub
	cp $< $@
	update-grub
configure_grub: $(GRUB_CONF)

.PHONY: iptables
$(IPTABLES_CONF):
	iptables -A FORWARD -s $(MGMT_SUBNET) -i eth1 -o eth0 -m conntrack --ctstate NEW -j ACCEPT
	iptables -A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
	iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
	sh -c "iptables-save > $@"
iptables: $(IPTABLES_CONF)

.PHONY: network
$(NETWORK_CONF): ./interfaces
	cp $< $@
	ifdown --exclude=lo -a && ifup --exclude=lo -a # Restart networking
network: $(NETWORK_CONF) iptables

.PHONY: sysctl
$(SYSCTL_CONF): ./sysctl.conf
	cp $< $@
	sysctl -p
sysctl: $(SYSCTL_CONF)

.PHONY: resolvconf
$(RESOLVCONF): ./resolv.conf.d_tail
	cp $< $@
	resolvconf -u
	restart network-manager
	ifdown --exclude=lo -a && ifup --exclude=lo -a # Restart networking
resolvconf: $(RESOLVCONF)

.PHONY: dnsmasq
$(DNSMASQ_CONF): ./dnsmasq.conf network
	cp $< $@
	service dnsmasq restart
dnsmasq: $(DNSMASQ_CONF)

.PHONY: configure_hosts
$(HOSTS_CONF): ../hosts network
	cp $< $@
configure_hosts: $(HOSTS_CONF)

.PHONY: configure_ethers
$(ETHERS_CONF): ./ethers network
	cp $< $@
configure_ethers: $(ETHERS_CONF)

.PHONY: configure_rclocal
$(RCLOCAL_CONF): ../rc.local
	cp $< $@

.PHONY: ssh
$(SSHD_CONF): ./sshd_config
	cp $< $@
$(SSH_CONF): ./ssh_config
	cp $< $@
ssh: $(SSHD_CONF) $(SSH_CONF) ../ssh_hosts network
	ssh-keyscan -t rsa,dsa,ecdsa -f ../ssh_hosts | sort -u - /etc/ssh/ssh_known_hosts | tee /etc/ssh/ssh_known_hosts
	restart ssh

.PHONY: nfs_server
nfs_server: network
	apt-get install -y nfs-kernel-server
	mkdir -p /export/home
	chmod 777 /export
	chmod 777 /export/home
	grep -q "/home"        /etc/fstab   || echo "/home           /export/home    none    bind            0       0" >> /etc/fstab
	grep -q "/export/home" /etc/exports || printf "# Note: no_root_squash is not a great idea, but is needed to allow clients to 'emacs blah'\n    /export/home 169.254.128.0/24(rw,nohide,insecure,no_subtree_check,async,no_root_squash)" >> /etc/exports
	service nfs-kernel-server restart

.PHONY: nfs_client
nfs_client: network
	apt-get install -y nfs-common
	grep -q "adp:/export/home" /etc/fstab || echo "adp:/export/home	/home			nfs	auto	0 0" >> /etc/fstab
	mount -a -t nfs

.PHONY: ntp
$(NTP_CONF): ../ntp.conf network
	/usr/sbin/ntpdate $(LOCAL_NTP_SERVER) ntp.ubuntu.com pool.ntp.org # Initial estimate
	apt-get install -y ntp
	cp $< $@
	restart ntp
	ntpq -p
ntp: $(NTP_CONF_FILE)

.PHONY: dev_ttyUSB0
dev_ttyUSB0:
	usermod -a -G dialout adp # Allow access to /dev/ttyUSB0

# Note that setsockopt(SO_SND/RCVBUF) actually allocates double the requested amount
.PHONY: socket_buffers
$(SOCK_WMEM_CONF):
	echo $(SOCK_BUF_LIMIT) | sudo tee $@
$(SOCK_RMEM_CONF):
	echo $(SOCK_BUF_LIMIT) | sudo tee $@
socket_buffers: $(SOCK_WMEM_CONF) $(SOCK_RMEM_CONF)

.PHONY: irq_affinity
irq_affinity:
	/etc/init.d/irqbalance stop
	cp ../irqbalance $(IRQBALANCE_CONF)
	cp ../configure_irq_affinity.py /usr/local/bin/
	grep -q configure_irq_affinity /etc/rc.local || sed -i '/^exit 0/i/usr/local/bin/configure_irq_affinity.py $(DATA_NETWORK_IFACE)' /etc/rc.local
	../configure_irq_affinity.py $(DATA_NETWORK_IFACE)
