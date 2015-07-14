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
import satelliteDebug as satDebug

try:
    import debug
except:
    pass

class SatelliteTask(pipeBase.CmdLineTask):
    _DefaultName = 'satellite'
    ConfigClass = pexConfig.Config


    @pipeBase.timeMethod
    def run(self, dataRef):

        import lsstDebug
        dbg = lsstDebug.Info(__name__).dbg

        v,c = dataRef.dataId['visit'], dataRef.dataId['ccd']
        
        #self.log.info("lsstDebug.Info(%s).debug = %s" % (str(__name__), str(dbg)))
        self.log.info("Detecting satellite trails in visit=%d, ccd=%d)" % (v,c))
        
        exposure = dataRef.get('calexp', immediate=True)

        if dbg:
            basedir = os.environ.get('SATELLITE_DATA')
            if basedir:
                basedir = os.path.join(os.environ.get("PWD"), "data")
            path = os.path.join(basedir, "%04d" %(v))
            try:
                os.mkdir(path)
            except:
                pass


        # run for regular satellites
        trailsSat = self.runSatellite(exposure, bins=4)
        if dbg:
            self.log.info("DEBUGGING: Now plotting SATELLITE detections.")
            filename = os.path.join(path,"satdebug-%05d-%03d.png" % (v, c))
            satDebug.debugPlot(self.finder, filename)

        # run for broad linear (aircraft?) features by binning
        trailsAc = self.runSatellite(exposure, bins=4, broadTrail=True)
        if dbg:
            self.log.info("DEBUGGING: Now plotting AIRCRAFT detections.")
            filename = os.path.join(path,"acdebug-%05d-%03d.png" % (v, c))
            satDebug.debugPlot(self.finder, filename)

        trails = trailsSat.merge(trailsAc)

        listMsg = "(%s,%s) Detected %d trail(s).  %s" % (v, c, len(trails), trails)
        self.log.info(listMsg)

        trailMsgs = []
        for i, trail in enumerate(trails):
            maskedPixels = trail.setMask(exposure)
            msg = "(%s,%s) Trail %d/%d %s:  maskPix: %d" % (v, c, i+1, len(trails), trail, maskedPixels)
            self.log.info(msg)
            trailMsgs.append(msg)
        
        if dbg:
            logfile = os.path.join(path, "log%05d-%03d.txt" % (v,c))
            with open(logfile, 'w') as log:
                log.write(listMsg+"\n")
                for msg in trailMsgs:
                    log.write(msg+'\n')

            exposure.writeFits(os.path.join(path,"exp%04d-%03d.fits"%(v,c)))
        
        return trails
        
    @pipeBase.timeMethod
    def runSatellite(self, exposure, bins=None, broadTrail=False):
            
        if broadTrail:
            luminosityLimit = 0.06 # low cut on pixel flux
            luminosityMax   = 50.0
            maskNPsfSigma   = 3.0*bins
            centerLimit     = 1.0   # about 1 pixel
            eRange          = 0.04  # about +/- 0.1
            houghBins       = 256   # number of r,theta bins (i.e. 256x256)
            kernelSigma     = 21    # pixels
            kernelWidth     = 41   # pixels
            widths          = [40.0, 90.0]  #width of an out of focus aircraft (unbinned)
            houghThresh     = 40    # counts in a r,theta bins
            skewLimit       = 120.0
            bLimit          = 1.0
        else:
            luminosityLimit = 0.05   # low cut on pixel flux
            luminosityMax   = 4.0e2 # max luminsity for pixel flux
            maskNPsfSigma   = 7.0
            centerLimit     = 0.5  # about 1 pixel
            eRange          = 0.02  # about +/- 0.1
            houghBins       = 256   # number of r,theta bins (i.e. 256x256)
            kernelSigma     = 9   # pixels
            kernelWidth     = 17   # pixels
            widths          = [1.0, 10.0]
            houghThresh     = 40    # counts in a r,theta bins
            skewLimit       = 20.0
            bLimit          = 0.6

        self.finder = satell.SatelliteFinder(
            kernelSigma=kernelSigma,
            kernelWidth=kernelWidth,
            bins=bins,
            centerLimit=centerLimit,
            eRange=eRange,
            houghThresh=houghThresh,
            houghBins=houghBins,
            luminosityLimit=luminosityLimit,
            luminosityMax=luminosityMax,
            skewLimit=skewLimit,
            bLimit=bLimit,
            log=self.log
        )

        trails = self.finder.getTrails(exposure, widths)
        return trails
        
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None



class SatelliteDistribRunner(pipeBase.TaskRunner):

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):

        allRefs = parsedCmd.id.refList
        nVisit = int((len(allRefs))**0.5)
        if nVisit**2 != len(allRefs):
            msg = "Mismatch in visits and ccds ... should be same length"
            raise ValueError(msg)
        targets = []
        for i in range(nVisit):
            ref = allRefs[i + i*nVisit]
            targets += [(ref, kwargs)]
        return targets
            
        
class SatelliteDistribTask(SatelliteTask):
    _DefaultName = 'distrib'
    ConfigClass = pexConfig.Config
    RunnerClass = SatelliteDistribRunner



    
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
    
