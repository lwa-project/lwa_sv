#!/bin/bash

ls /home/adp/log/*.gz | xargs -n1 /home/adp/lwa_sv/scripts/uploadLogfile.py
