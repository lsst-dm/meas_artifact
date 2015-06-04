#!/usr/bin/env python
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

import satellite as satell

class SatelliteTask(pipeBase.CmdLineTask):
    _DefaultName = 'satellite'
    ConfigClass = pexConfig.Config()

    def run(self, dataRef):
        exposure = dataRef.get('calexp')

        kernelSigma = 9    # pixels
        kernelSize  = 31   # pixels
        centerLimit = 1.0  # about 1 pixel
        ellipLimit  = 0.1  # about +/- 0.1
        
        houghLimit      = 40    # counts in a r,theta bins
        houghBins       = 256   # number of r,theta bins (i.e. 256x256)
        luminosityLimit = 0.02  # low cut on pixel flux
        
        finder = satell.SatelliteFinder(
            kernelSigma=kernelSigma,
            kernelSize=kernelSize,
            centerLimit=centerLimit,
            eLimit=eLimit,
            houghLimit=houghLimit,
            houghBins=houghBins,
            luminosityLimit=luminosityLimit
        )

        satelliteTrails = finder.getTrails()

        for trail in satelliteTrails:
            trail.setMask(exposure, maskPlane)

        
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None

if __name__ == '__main__':
    SatelliteTask.parseAndRun()
    
