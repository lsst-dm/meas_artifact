#!/usr/bin/env python

import sys
from lsst.meas.artifact import PoolSatelliteTask
PoolSatelliteTask.parseAndSubmit(sys.argv[1:])
