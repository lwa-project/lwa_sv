
# TODO: python-simplejson

INSTALL_BIN_DIR           ?= /usr/local/bin
SERVICE_CONF_DIR          ?= /etc/init

all:
	@echo "Please run either 'make configure/install_headnode/server'"

configure_headnode:
	$(MAKE) -C ./config/ configure_headnode
.PHONY: configure_headnode
configure_server:
	$(MAKE) -C ./config/ configure_server
.PHONY: configure_server

install_headnode: install_adp_control_service
install_server:   cuda install_adp_pipeline

$(INSTALL_BIN_DIR)/adp_control.py: scripts/adp_control.py
	cp $< $@
$(SERVICE_CONF_DIR)/adp-control.conf: config/headnode/adp-control.conf
	cp $< $@
install_adp_control_service: $(INSTALL_BIN_DIR)/adp_control.py $(SERVICE_CONF_DIR)/adp-control.conf
	initctl reload-configuration
	start adp-control
.PHONY: install_adp_control_service

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
