#!/usr/bin/env python

import sys
from lsst.meas.satellite.satelliteTask import HoughSatelliteTask as SatelliteTask
SatelliteTask.parseAndRun(sys.argv[1:])
