#!/usr/bin/env python

import sys, os
import math
import collections
import time
import cPickle as pickle
import numpy as np

import lsst.afw.cameraGeom as afwCg
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

import hsc.pipe.base.butler as hscButler
import hsc.pipe.base.pool as basePool
import hsc.pipe.base.parallel as basePara

basePool.Debugger().enabled = True

import satelliteFinder as satFind
import satelliteDebug  as satDebug
import satelliteTrail  as satTrail

try:
    import debug
except:
    pass


class SatelliteRunner(pipeBase.TaskRunner):

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs['detectionType'] = parsedCmd.detectionType
        return [(ref, kwargs) for ref in parsedCmd.id.refList]
    
    
class SatelliteTask(pipeBase.CmdLineTask):
    _DefaultName = 'satellite'
    ConfigClass = pexConfig.Config
    RunnerClass = SatelliteRunner

    @classmethod
    def _makeArgumentParser(cls):
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "calexp", help="Data ID, e.g. --id tract=1234 patch=2,2",
                               ContainerClass=pipeBase.DataIdContainer)
        parser.add_argument("-d", "--detectionType",
                            choices=('sat', 'ac', 'all'), default='all', help="Set to run")
        return parser

    
    @pipeBase.timeMethod
    def run(self, dataRef, **kwargs):

        detectionType = kwargs.get('detectionType', 'all')
        
        import lsstDebug
        dbg = lsstDebug.Info(__name__).dbg

        v,c = dataRef.dataId['visit'], dataRef.dataId['ccd']
        
        #self.log.info("lsstDebug.Info(%s).debug = %s" % (str(__name__), str(dbg)))
        self.log.info("Detecting satellite trails in visit=%d, ccd=%d)" % (v,c))
        
        exposure = dataRef.get('calexp', immediate=True)

        basedir = os.environ.get('SATELLITE_DATA')
        if basedir:
            basedir = os.path.join(os.environ.get("PWD"), "data")
        path = os.path.join(basedir, "%04d" %(v))
        try:
            os.mkdir(path)
        except:
            pass

        t0 = time.time()

        coord1, coord2 = False, False 

        trails = satTrail.SatelliteTrailList(0.0, 0.0, 0.0)
        
        # run for regular satellites
        if detectionType in ('all', 'sat'):
            trailsSat = self.runSatellite(exposure, bins=4)
            if dbg:
                self.log.info("DEBUGGING: Now plotting SATELLITE detections.")
                if coord1:
                    filename = os.path.join(path,"coord-%05d-%03d.png" % (v,c))
                    satDebug.coordPlot(exposure, self.finder, filename)
                    sys.exit()
                filename = os.path.join(path,"satdebug-%05d-%03d.png" % (v, c))
                satDebug.debugPlot(self.finder, filename)
            print trailsSat
            trails = trailsSat.merge(trails, drMax=90.0, dThetaMax=0.15)
            
        # run for broad linear (aircraft?) features by binning
        if detectionType in ('all', 'ac'):
            trailsAc = self.runSatellite(exposure, bins=8, broadTrail=True)
            if dbg:
                self.log.info("DEBUGGING: Now plotting AIRCRAFT detections.")
                if coord2:
                    filename = os.path.join(path,"coord-%05d-%03d.png" % (v,c))
                    satDebug.coordPlot(exposure, self.finder, filename)
                    sys.exit()
                filename = os.path.join(path,"acdebug-%05d-%03d.png" % (v, c))
                satDebug.debugPlot(self.finder, filename)
            print trailsAc
            trails = trailsAc.merge(trails, drMax=90.0, dThetaMax=0.15)
            
        
        if True:
            picfile = os.path.join(path, "trails%05d-%03d.pickle" % (v,c))
            with open(picfile, 'w') as fp:
                bundle = ((v,c), trails, time.time() - t0)
                pickle.dump(bundle, fp)

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
        
        return trails, time.time() - t0
        
    @pipeBase.timeMethod
    def runSatellite(self, exposure, bins=None, broadTrail=False):
            
        if broadTrail:
            luminosityLimit = 0.02 # low cut on pixel flux
            maskNPsfSigma   = 3.0*bins
            centerLimit     = 1.0           # about 1 pixel
            eRange          = 0.08          # about +/- 0.1
            houghBins       = 200           # number of r,theta bins (i.e. 256x256)
            kernelSigma     = 9 #13            # pixels
            kernelWidth     = 15 #29           # pixels
            widths          = [40.0, 70.0, 100]  # width of an out of focus aircraft (unbinned)
            houghThresh     = 40            # counts in a r,theta bins
            skewLimit       = 50.0 #400.0
            bLimit          = 1.5 #3.0
            maxTrailWidth   = 1.6*bins
        else:
            luminosityLimit = 0.02   # low cut on pixel flux
            maskNPsfSigma   = 7.0
            centerLimit     = 1.2  # about 1 pixel
            eRange          = 0.08  # about +/- 0.1
            houghBins       = 200   # number of r,theta bins (i.e. 256x256)
            kernelSigma     = 7   # pixels
            kernelWidth     = 11   # pixels
            widths          = [1.0, 8.0]
            houghThresh     = 40    # counts in a r,theta bins
            skewLimit       = 10.0
            bLimit          = 1.4
            maxTrailWidth   = 2.1*bins

        self.finder = satFind.SatelliteFinder(
            kernelSigma=kernelSigma,
            kernelWidth=kernelWidth,
            bins=bins,
            centerLimit=centerLimit,
            eRange=eRange,
            houghThresh=houghThresh,
            houghBins=houghBins,
            luminosityLimit=luminosityLimit,
            skewLimit=skewLimit,
            bLimit=bLimit,
            maxTrailWidth=maxTrailWidth,
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
    
