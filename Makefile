
# TODO: Install pythons modules: python-simplejson, paramiko, pyzmq

INSTALL_BIN_DIR           ?= /usr/local/bin
INSTALL_SHARE_DIR         ?= /usr/local/share/adp
SERVICE_CONF_DIR          ?= /etc/init
#SERVICE_LOG_DIR           ?= /var/log

all:
	@echo "Please run either 'make configure/install_headnode/server'"

configure_headnode:
	$(MAKE) -C ./config/ configure_headnode
.PHONY: configure_headnode
configure_server:
	$(MAKE) -C ./config/ configure_server
.PHONY: configure_server

configure_server_partial:
	$(MAKE) -C ./config/ configure_server_partial
.PHONY: configure_server_partial

install_headnode: install_adp_control_service install_adp_tengine_service
#install_server:   cuda install_adp_pipeline
install_server:   install_adp_pipeline

$(INSTALL_BIN_DIR)/adp/: scripts/adp/
	test -d $(INSTALL_BIN_DIR)/adp || mkdir $(INSTALL_BIN_DIR)/adp
	cp $<* $@
$(INSTALL_BIN_DIR)/adp_control.py: scripts/adp_control.py $(INSTALL_BIN_DIR)/adp/
	cp $< $@
$(SERVICE_CONF_DIR)/adp-control.conf: config/headnode/adp-control.conf
	cp $< $@
$(INSTALL_BIN_DIR)/adp_tbn.py: scripts/adp_tbn.py $(INSTALL_BIN_DIR)/adp/
	cp $< $@
$(SERVICE_CONF_DIR)/adp-tbn.conf: config/servers/adp-tbn.conf
	cp $< $@
$(INSTALL_SHARE_DIR)/adp_config.json: config/adp_config.json
	test -d $(INSTALL_SHARE_DIR) || mkdir $(INSTALL_SHARE_DIR)
	cp $< $@
install_adp_control_service: $(INSTALL_BIN_DIR)/adp_control.py $(SERVICE_CONF_DIR)/adp-control.conf $(INSTALL_SHARE_DIR)/adp_config.json install_sshpass
	initctl reload-configuration
	start adp-control
	tail -n 40 /var/log/upstart/adp-control.log
.PHONY: install_adp_control_service

install_adp_tengine_service: $(INSTALL_BIN_DIR)/adp_tengine.py $(SERVICE_CONF_DIR)/adp-tengine-0.conf $(SERVICE_CONF_DIR)/adp-tengine-1.conf $(INSTALL_SHARE_DIR)/adp_config.json install_numactl
        initctl reload-configuration
	stop adp-tengine-0 ; start adp-tengine-0
	stop adp-tengine-1 ; start adp-tengine-1
.PHONY: install_adp_tengine_service

install_adp_tbn_service: $(INSTALL_BIN_DIR)/adp_tbn.py $(SERVICE_CONF_DIR)/adp-tbn.conf $(INSTALL_SHARE_DIR)/adp_config.json
	initctl reload-configuration
	stop  adp-tbn ; start adp-tbn # Note: Need to stop+start to reload the conf file
.PHONY: install_adp_tbn_service

install_adp_drx_service: $(INSTALL_BIN_DIR)/adp_drx.py $(SERVICE_CONF_DIR)/adp-drx-0.conf $(SERVICE_CONF_DIR)/adp-drx-1.conf $(INSTALL_SHARE_DIR)/adp_config.json
        initctl reload-configuration
	stop  adp-drx-0 ; start adp-drx-0
	stop  adp-drx-1 ; start adp-drx-1
.PHONY: install_adp_drx_service

install_adp_pipeline: install_numactl install_adp_tbn_service install_adp_drx_service
.PHONY: install_adp_pipeline

install_numactl:
	apt-get install -y numactl
.PHONY: install_numactl

install_sshpass:
	apt-get install -y sshpass
.PHONY: install_sshpass

CUDA_DOWNLOAD_PATH        ?= "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb"
CUDA_INSTALL_FILE         ?= cuda-repo-ubuntu1404_7.0-28_amd64.deb
CUDA_VERSION              ?= cuda-7.0
CUDA_DIR                  ?= /usr/local/$(CUDA_VERSION)
NVCC                      ?= $(CUDA_DIR)/bin/nvcc

$(CUDA_INSTALL_FILE):
	wget $(CUDA_DOWNLOAD_PATH) -o $@
$(NVCC): $(CUDA_INSTALL_FILE)
	dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
	apt-get update
	apt-get install -y cuda-7-0
cuda: $(NVCC)
.PHONY: cuda
