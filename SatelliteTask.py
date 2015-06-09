#!/usr/bin/env python
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

import satellite as satell

class SatelliteTask(pipeBase.CmdLineTask):
    _DefaultName = 'satellite'
    ConfigClass = pexConfig.Config

    def run(self, dataRef):

        self.log.info("Detecting satellite trails in %s" % (str(dataRef.dataId)))
        
        exposure = dataRef.get('calexp', immediate=True)

        # run for regular satellites
        self.runSatellite(exposure)
        # run for broad linear (aircraft?) features by binning
        self.runSatellite(exposure, bins=16)

        exposure.writeFits("exp%04d-%03d.fits"%(dataRef.dataId['visit'], dataRef.dataId['ccd']))

        
    def runSatellite(self, exposure, bins=None):
        if bins:
            exp = type(exposure)(afwMath.binImage(exposure.getMaskedImage(), bins))
            exp.setMetadata(exposure.getMetadata())
            exp.setPsf(exposure.getPsf())
            luminosityLimit = 1.0   # low cut on pixel flux
            luminosityMax = 2.0
            maskNPsfSigma = 3.0*bins
        else:
            exp = exposure
            luminosityLimit = 4.0   # low cut on pixel flux
            luminosityMax   = 10.0  # max luminsity for pixel flux
            maskNPsfSigma = 7.0

        kernelSigma = 11    # pixels
        kernelSize  = 21   # pixels
        centerLimit = 1.0  # about 1 pixel
        eRange      = 0.1  # about +/- 0.1
        
        houghThresh     = 40    # counts in a r,theta bins
        houghBins       = 256   # number of r,theta bins (i.e. 256x256)
        
        finder = satell.SatelliteFinder(
            kernelSigma=kernelSigma,
            kernelSize=kernelSize,
            centerLimit=centerLimit,
            eRange=eRange,
            houghThresh=houghThresh,
            houghBins=houghBins,
            luminosityLimit=luminosityLimit,
            luminosityMax=luminosityMax
        )

        satelliteTrails = finder.getTrails(exp)
        self.log.info("Detected %d satellite trails.  cand-pix=%d bin-max=%d" %
                      (len(satelliteTrails), satelliteTrails.nTotal, satelliteTrails.binMax))
        
        for i, trail in enumerate(satelliteTrails):
            if bins:
                trail.r *= bins
            maskedPixels = trail.setMask(exposure, nSigma=maskNPsfSigma)
            self.log.info("Trail %d of %d (r=%.1f,theta=%.4f):  cand-pix=%d max-bin-count=%d mask-pix=%d" %
                          (i+1, len(satelliteTrails), trail.r, trail.theta,
                           trail.nAboveThresh, trail.houghBinMax, maskedPixels))

        
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None

if __name__ == '__main__':
    SatelliteTask.parseAndRun()
    
