#!/usr/bin/env python

import sys
from SatelliteTask import PoolSatelliteTask
PoolSatelliteTask.parseAndSubmit(sys.argv[1:])
