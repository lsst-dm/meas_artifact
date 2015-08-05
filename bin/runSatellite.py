#!/usr/bin/env python

import sys
from lsst.meas.artifact import HoughSatelliteTask as SatelliteTask
SatelliteTask.parseAndRun(sys.argv[1:])
