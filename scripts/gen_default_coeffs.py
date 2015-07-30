#!/usr/bin/env python

import numpy as np

if __name__ == "__main__":
	import sys
	data = np.ones((16,32), dtype='>h').tobytes()
	sys.stdout.write(data)
