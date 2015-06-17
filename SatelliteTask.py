#!/usr/bin/env python
import os, math, collections
import lsst.afw.cameraGeom as afwCg
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

import hsc.pipe.base.butler as hscButler

import hsc.pipe.base.pool as basePool
import hsc.pipe.base.parallel as basePara

basePool.Debugger().enabled = True

import numpy as np
#np.seterr(all='raise')

import satellite as satell

class SatelliteTask(pipeBase.CmdLineTask):
    _DefaultName = 'satellite'
    ConfigClass = pexConfig.Config

    def run(self, dataRef):

        self.log.info("Detecting satellite trails in %s" % (str(dataRef.dataId)))
        
        exposure = dataRef.get('calexp', immediate=True)
        v,c = dataRef.dataId['visit'], dataRef.dataId['ccd']
        basedir = os.environ.get('SATELLITE_DATA', '/home/bick/sandbox/hough/data')
        path = os.path.join(basedir, "%04d" %(v))
        try:
            os.mkdir(path)
        except:
            pass
        logfile = os.path.join(path, "log%05d-%03d.txt" % (v,c))
        with open(logfile, 'w') as log:
            # run for regular satellites
            self.runSatellite(exposure, bins=2, log=log)
            # run for broad linear (aircraft?) features by binning
            #self.runSatellite(exposure, bins=4, broadTrail=True, log=log)

        exposure.writeFits(os.path.join(path,"exp%04d-%03d.fits"%(v,c)))

        
    def runSatellite(self, exposure, bins=None, broadTrail=False, log=None):
            
        if broadTrail:
            luminosityLimit = 0.01   # low cut on pixel flux
            luminosityMax = 4.0
            maskNPsfSigma = 3.0*bins
            centerLimit = 2.0   # about 1 pixel
            eRange      = 0.04  # about +/- 0.1
            houghBins      = 128   # number of r,theta bins (i.e. 256x256)
            kernelSigma = 21    # pixels
            kernelSize  = 41   # pixels
            width       = 60.0 #100.0  #width of an out of focus aircraft (unbinned)
            houghThresh     = 40    # counts in a r,theta bins
            skewLimit   = 200.0
            widthToPsfLimit = 0.2
        else:
            luminosityLimit = 0.2   # low cut on pixel flux
            luminosityMax   = 120.0  # max luminsity for pixel flux
            maskNPsfSigma = 7.0
            centerLimit = 0.5  # about 1 pixel
            eRange      = 0.025  # about +/- 0.1
            houghBins       = 256   # number of r,theta bins (i.e. 256x256)
            kernelSigma = 9    # pixels
            kernelSize  = 17   # pixels
            width=None
            houghThresh     = 10    # counts in a r,theta bins
            skewLimit       = 20.0
            widthToPsfLimit = 0.1
        
        finder = satell.SatelliteFinder(
            kernelSigma=kernelSigma,
            kernelSize=kernelSize,
            centerLimit=centerLimit,
            eRange=eRange,
            houghThresh=houghThresh,
            houghBins=houghBins,
            luminosityLimit=luminosityLimit,
            luminosityMax=luminosityMax,
            skewLimit=skewLimit,
            widthToPsfLimit=widthToPsfLimit
        )

        trails = finder.getTrails(exposure, bins=bins, width=width)
        msg = ""
        if bins:
            msg = "(binned %dx) " % bins
        msg += "Detected %d satellite trails.  cand-pix: %d bin-max: %d  psfSigma: %.2f" % \
               (len(trails), trails.nTotal, trails.binMax, trails.psfSigma)
        self.log.info(msg)
        if log:
            log.write(msg+"\n")
        
        for i, trail in enumerate(trails):
            maskedPixels = trail.setMask(exposure, nSigma=maskNPsfSigma)
            msg = "Trail %d of %d (r: %.1f,theta: %.4f):  cand-pix: %d max-bin-count: %d mask-pix: %d" % \
                  (i+1, len(trails), trail.r, trail.theta,
                   trail.nAboveThresh, trail.houghBinMax, maskedPixels)          
            self.log.info(msg)
            if log:
                log.write(msg+"\n")
        
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None








class PoolSatelliteConfig(pexConfig.Config):
    satellite    = pexConfig.ConfigurableField(target=SatelliteTask, doc="satellite")

    
class PoolSatelliteTask(basePara.BatchPoolTask):
    RunnerClass = hscButler.ButlerTaskRunner
    ConfigClass = PoolSatelliteConfig
    _DefaultName = "poolSatellite"

    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None

    def __init__(self, *args, **kwargs):
        super(PoolSatelliteTask, self).__init__(*args, **kwargs)
        self.makeSubtask("satellite")
 

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numNodes, numProcs):
        numCcds = sum(1 for raft in parsedCmd.butler.get("camera") for ccd in afwCg.cast_Raft(raft))
        numCycles = int(math.ceil(numCcds/float(numNodes*numProcs)))
        numExps = len(cls.RunnerClass.getTargetList(parsedCmd))
        return time*numExps*numCycles

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        doBatch = kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", level="visit", ContainerClass=pipeBase.DataIdContainer,
                               help="data ID, e.g. --id visit=12345")
        return parser

    @basePool.abortOnError
    def run(self, expRef, butler):
        pool = basePool.Pool("poolSatellite")
        pool.cacheClear()
        pool.storeSet(butler=butler)

        self.log.info("Beginning processing of ExpRef: " + str(expRef.dataId))

        dataIdList = dict([(ccdRef.get("ccdExposureId"), ccdRef.dataId)
                           for ccdRef in expRef.subItems("ccd") if ccdRef.datasetExists("src")])
        dataIdList = collections.OrderedDict(sorted(dataIdList.items()))

        self.log.info("DataIdList to be processed: " + str( dataIdList.values()))

        # Scatter: run each CCD separately
        structList = pool.map(self.process, dataIdList.values())

        
    def process(self, cache, dataId):
        dataRef = hscButler.getDataRef(cache.butler, dataId, datasetType="src")
        ccdId = dataRef.get("ccdExposureId")

        with self.logOperation("Started satellite %s (ccdId=%d) on %s" % (dataId, ccdId, basePool.NODE)):
            try:
                result = self.satellite.run(dataRef)
            except Exception, e:
                self.log.warn("Satellite failed %s: %s\n" % (dataId, e))
                import traceback
                traceback.print_exc()
                return None

            if result is not None:
                # Cache the results (in particular, the image)
                cache.result = result

            self.log.info("Finished satellite %s (ccdId=%d) on %s" % (dataId, ccdId, basePool.NODE))
            return pipeBase.Struct(ccdId=ccdId)


        
if __name__ == '__main__':
    SatelliteTask.parseAndRun()
    
