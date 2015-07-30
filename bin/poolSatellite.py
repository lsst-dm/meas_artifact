#!/usr/bin/env python

import sys
from satellite.satelliteTask import PoolSatelliteTask
PoolSatelliteTask.parseAndSubmit(sys.argv[1:])
