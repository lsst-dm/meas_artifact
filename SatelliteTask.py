#!/usr/bin/env python
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

import satellite as satell

class SatelliteTask(pipeBase.CmdLineTask):
    _DefaultName = 'satellite'
    ConfigClass = pexConfig.Config

    def run(self, dataRef):
        exposure = dataRef.get('calexp')

        self.log.info("Detecting satellite trails in %s" % (str(dataRef.dataId)))
        kernelSigma = 11    # pixels
        kernelSize  = 21   # pixels
        centerLimit = 1.0  # about 1 pixel
        eRange      = 0.1  # about +/- 0.1
        
        houghThresh     = 40    # counts in a r,theta bins
        houghBins       = 256   # number of r,theta bins (i.e. 256x256)
        luminosityLimit = 4.0   # low cut on pixel flux
        
        finder = satell.SatelliteFinder(
            kernelSigma=kernelSigma,
            kernelSize=kernelSize,
            centerLimit=centerLimit,
            eRange=eRange,
            houghThresh=houghThresh,
            houghBins=houghBins,
            luminosityLimit=luminosityLimit
        )

        satelliteTrails = finder.getTrails(exposure)
        self.log.info("Detected %d satellite trails.  cand-pix=%d bin-max=%d" %
                      (len(satelliteTrails), satelliteTrails.nTotal, satelliteTrails.binMax))
        
        for i, trail in enumerate(satelliteTrails):
            maskedPixels = trail.setMask(exposure)
            self.log.info("Trail %d of %d (r=%.1f,theta=%.4f):  cand-pix=%d max-bin-count=%d mask-pix=%d" %
                          (i+1, len(satelliteTrails), trail.r, trail.theta,
                           trail.nAboveThresh, trail.houghBinMax, maskedPixels))

        exposure.writeFits("exp%04d-%03d.fits"%(dataRef.dataId['visit'], dataRef.dataId['ccd']))
        
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None

if __name__ == '__main__':
    SatelliteTask.parseAndRun()
    
