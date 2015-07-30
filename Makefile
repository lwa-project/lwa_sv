
CUDA_DOWNLOAD_PATH        ?= "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb"
CUDA_INSTALL_FILE         ?= cuda-repo-ubuntu1404_7.0-28_amd64.deb
CUDA_VERSION              ?= cuda-7.0
CUDA_DIR                  ?= /usr/local/$(CUDA_VERSION)
NVCC                      ?= $(CUDA_DIR)/bin/nvcc
SOCK_WRITE_BUF_LIMIT_FILE ?= /proc/sys/net/core/wmem_max
SOCK_READ_BUF_LIMIT_FILE  ?= /proc/sys/net/core/rmem_max
LOCAL_NTP_SERVER          ?= cfa-ntp.cfa.harvard.edu
NTP_CONF                  ?= /etc/ntp.conf
INSTALL_BIN_DIR           ?= /usr/local/bin
SERVICE_CONF_DIR          ?= /etc/init

SOCK_BUF_LIMIT ?= "536870912"

$(CUDA_INSTALL_FILE):
	wget $(CUDA_DOWNLOAD_PATH) -o $@
$(SOCK_WRITE_BUF_LIMIT_FILE):
	echo $(SOCK_BUF_LIMIT) | tee $@
$(SOCK_READ_BUF_LIMIT_FILE):
	echo $(SOCK_BUF_LIMIT) | tee $@

$(NVCC): $(CUDA_INSTALL_FILE)
	dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
	apt-get update
	apt-get install -y cuda-7-0

$(NTP_CONF_FILE): headnode/ntp.conf
	/usr/sbin/ntpdate ntp.ubuntu.com pool.ntp.org $(LOCAL_NTP_SERVER)
	apt-get install -y ntp
	cp $< $@
	echo server $(LOCAL_NTP_SERVER) | tee -a $@
	restart ntp
	ntpq -p
ntpdaemon: $(NTP_CONF_FILE)
.PHONY: ntpdaemon

$(INSTALL_BIN_DIR)/adp_control.py: scripts/adp_control.py
	cp $< $@
$(SERVICE_CONF_DIR)/adp-control.conf: headnode/adp-control.conf
	cp $< $@
adp_control_service: $(INSTALL_BIN_DIR)/adp_control.py $(SERVICE_CONF_DIR)/adp-control.conf
	initctl reload-configuration
	start adp-control
.PHONY: adp_control_service

# TODO: denyhosts
# TODO: IPv4 forwarding
# TODO: dnsmasq
configure_headnode: headnode/grub headnode/interfaces headnode/sshd_config
	cp headnode/grub               /etc/default/
	update-grub
	cp headnode/interfaces         /etc/network/
	ifdown --exclude=lo -a && ifup --exclude=lo -a # Restart networking
	cp headnode/sshd_config        /etc/ssh/
	cp headnode/ssh_known_hosts     /etc/ssh/
	restart ssh
	cp headnode/resolv.conf.d_base /etc/resolvconf/resolv.conf.d/
	resolvconf -u
	apt-get install -y nfs-kernel-server
	mkdir -p /export/home
	chmod 777 /export
	chmod 777 /export/home
	grep -q "/home" /etc/fstab || echo "/home           /export/home    none    bind            0       0" >> /etc/fstab
	grep -q "/export/home" /etc/exports || printf "# Note: no_root_squash is not a great idea, but is needed to allow clients to 'emacs blah'\n    /export/home 169.254.128.0/24(rw,nohide,insecure,no_subtree_check,async,no_root_squash)" >> /etc/exports
.PHONY: configure_headnode

configure_server: servers/grub servers/interfaces servers/sshd_config
	cp servers/grub            /etc/default/
	update-grub
	cp servers/interfaces      /etc/network/
	ifdown --exclude=lo -a && ifup --exclude=lo -a # Restart networking
	cp servers/sshd_config     /etc/ssh/
	cp servers/ssh_known_hosts /etc/ssh/
	restart ssh
	cp servers/resolv.conf.d_base /etc/resolvconf/resolv.conf.d/base
	resolvconf -u
	apt-get install -y nfs-common
	grep -q "adp:/export/home" /etc/fstab || echo "adp:/export/home	/home			nfs	auto	0 0" >> /etc/fstab
	mount -a -t nfs
.PHONY: configure_server

install_headnode: ntpdaemon adp_control_service
.PHONY: install_headnode

server: $(SOCK_WRITE_BUF_LIMIT_FILE) $(SOCK_READ_BUF_LIMIT_FILE)

