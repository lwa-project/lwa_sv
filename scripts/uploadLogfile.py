#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import requests
from socket import gethostname


URL = "https://lda10g.unm.edu/metadata/sorter/index.py"
KEY = "c0843461abe746a4608dd9c897f9b261"
SITE = "lwasv"
TYPE = "SSLOG"

# Send the update to lwalab
r = os.path.realpath(sys.argv[1])
f = requests.post(URL,
                  data={'key': KEY, 'site': SITE, 'type': TYPE, 'subsystem': 'ADP'},
                  files={'file': open(r, 'rb')},
                  verify=False) # We don't have a certiticate for lda10g.unm.edu
print(f.text)
f.c
