#!/bin/bash

#
# Host validation
#

if [ `hostname` != "adp" ]; then
	echo "This script must be run on the head node"
	exit 1
fi

#
# Argument parsing
#

DO_CONFIG=1
DO_SOFTWARE=1
DO_UPSTART=1
DO_RESTART=0
DO_QUERY=0
while [[ $# -gt 0 ]]; do
	key="${1}"
	case ${key} in
		-h|--help)
			echo "syncSoftware.py - Script to help get the ADP software in sync"
			echo "                  across the various nodes."
			echo ""
			echo "Usage:"
			echo "sudo ./syncSoftware.py [OPTIONS]"
			echo ""
			echo "Options:"
			echo "-h,--help            Show this help message"
			echo "-c,--config-only     Only update the configuration file and restart the ADP services"
			echo "-s,--software-only   Only update the ADP software and restart the ADP services"
			echo "-u,--upstart-only    Only update the ADP systemd service definitions"
			echo "-r,--restart         Rrestart the ADP services after an update"
			echo "-o,--restart-only    Do not udpdate, only restart the ADP services"
			echo "-q,--query           Query the status of the ADP services"
			exit 0
			;;
		-c|--config-only)
			DO_CONFIG=1
			DO_SOFTWARE=0
			DO_UPSTART=0
			DO_QUERY=0
			;;
		-s|--software-only)
			DO_CONFIG=0
                        DO_SOFTWARE=1
			DO_UPSTART=0
			DO_QUERY=0
                        ;;
		-u|--upstart-only)
			DO_CONFIG=0
			DO_SOFTWARE=0
			DO_UPSTART=1
			DO_QUERY=0
			;;
		-r|--restart)
                        DO_RESTART=1
			DO_QUERY=0
                        ;;
		-0|--restart-only)
			DO_CONFIG=0
			DO_SOFTWARE=0
			DO_UPSTART=0
			DO_RESTART=1
			DO_QUERY=0
			;;
		-q|--query)
			DO_CONFIG=0
                        DO_SOFTWARE=0
			DO_UPSTART=0
                        DO_RESTART=0
			DO_QUERY=1
			;;
		*)
		;;
	esac
	shift
done

#
# Permission validation
#

if [ `whoami` != "root" ]; then
	echo "This script must be run with superuser privileges"
	exit 2
fi

#
# Configuration
#

if [ "${DO_CONFIG}" == "1" ]; then
	SRC_PATH=/home/adp/lwa_sv/config
	DST_PATH=/usr/local/share/adp
	
	for node in `seq 0 6`; do
		rsync -e ssh -avH ${SRC_PATH}/adp_config.json adp${node}:${DST_PATH}/
	done
fi


#
# Software
#

if [ "${DO_SOFTWARE}" == "1" ]; then
	SRC_PATH=/home/adp/lwa_sv/scripts
	DST_PATH=/usr/local/bin
	
	for node in `seq 0 6`; do
		if [ "${node}" == "0" ]; then
			rsync -e ssh -avH ${SRC_PATH}/adp ${SRC_PATH}/adp_control.py ${SRC_PATH}/adp_tengine.py ${SRC_PATH}/adp_enable_triggering.py adp${node}:${DST_PATH}/
		else
			rsync -e ssh -avH ${SRC_PATH}/adp ${SRC_PATH}/adp_tbn.py ${SRC_PATH}/adp_drx.py adp${node}:${DST_PATH}/
		fi
	done
fi


#
# Upstart
#

if [ "${DO_UPSTART}" == "1" ]; then
	SRC_PATH=/home/adp/lwa_sv/config
	DST_PATH=/etc/systemd/system/
	
	for node in `seq 0 6`; do
		if [ "${node}" == "0" ]; then
			rsync -e ssh -avH ${SRC_PATH}/headnode/adp-*.service adp${node}:${DST_PATH}/
		else
			rsync -e ssh -avH ${SRC_PATH}/servers/adp-*.service adp${node}:${DST_PATH}/
		fi
		ssh adp${node} "systemctl daemon-reload"
	done
fi

#
# Restart
#

if [ "${DO_RESTART}" == "1" ]; then
	for node in `seq 0 6`; do
		if [ "${node}" == "0" ]; then
			ssh adp${node} "restart adp-control && restart adp-tengine-0 && restart adp-tengine-1"
		else
			ssh adp${node} "restart adp-tbn && restart adp-drx-0 && restart adp-drx-1"
		fi
	done
fi

#
# Query
#

if [ "${DO_QUERY}" == "1" ]; then
        for node in `seq 0 6`; do
                if [ "${node}" == "0" ]; then
                        ssh adp${node} "status adp-control && status adp-tengine-0 && status adp-tengine-1"
                else
                        ssh adp${node} "status adp-tbn && status adp-drx-0 && status adp-drx-1"
                fi
        done
fi

