
all:
	@echo "Please run either 'make configure_headnode' or 'make configure_server'"

configure_headnode:
	$(MAKE) -C ./headnode/ configure
.PHONY: configure_headnode

configure_server:
	$(MAKE) -C ./servers/ configure
.PHONY: configure_server

configure_server_partial:
	$(MAKE) -C ./servers/ configure_partial
.PHONY: configure_server_partial

# Note: Can alternatively prefix commands with @ to silence echoing
#.SILENT:
