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



class SatelliteFinderConfig(pexConfig.Config):
    luminosityLimit = pexConfig.Field(dtype=float, default=0.02,doc="Lowest luminosity in Std.Dev.")
    centerLimit     = pexConfig.Field(dtype=float, default=1.2, doc="Max centroid value [pixels]")
    eRange          = pexConfig.Field(dtype=float, default=0.08,doc="Max ellipticity range.")
    skewLimit       = pexConfig.Field(dtype=float, default=10.0,doc="Max value of x-cor skew (3rd moment)")
    bLimit          = pexConfig.Field(dtype=float, default=1.4, doc="Max error in x-cor trail width.")
    kernelSigma     = pexConfig.Field(dtype=float, default=7.0, doc="Gauss sigma to use for x-cor kernel.")
    kernelWidth     = pexConfig.Field(dtype=int,   default=11,  doc="Width of x-cor kernel in pixels")
    
    houghBins       = pexConfig.Field(dtype=int,   default=200, doc="Number of bins to use in r,theta space.")
    houghThresh     = pexConfig.Field(dtype=int,   default=40,
                                      doc="Min number of 'hits' in a Hough bin for a detection.")
    maxTrailWidth   = pexConfig.Field(dtype=float, default=2.1,
                                      doc="Discard trails with measured widths greater than this (pixels).")
    widths          = pexConfig.ListField(dtype=float, default= (1.0, 8.0),
                                          doc="*unbinned* width of trail to search for.")
    bins            = pexConfig.Field(dtype=int,   default=4,   doc="How to bin the image before detection")

class SatelliteFinderTask(pipeBase.Task):
    _DefaultName = 'satelliteBase'
    ConfigClass = SatelliteFinderConfig


    def __init__(self, *args, **kwargs):
        super(SatelliteFinderTask,self).__init__(*args, **kwargs)
        self.finder = satFind.SatelliteFinder(
            kernelSigma     = self.config.kernelSigma,
            kernelWidth     = self.config.kernelWidth,
            bins            = self.config.bins,
            centerLimit     = self.config.centerLimit,
            eRange          = self.config.eRange,
            houghThresh     = self.config.houghThresh,
            houghBins       = self.config.houghBins,
            luminosityLimit = self.config.luminosityLimit,
            skewLimit       = self.config.skewLimit,
            bLimit          = self.config.bLimit,
            maxTrailWidth   = self.config.maxTrailWidth*self.config.bins, # correct for binning
            log             = self.log
        )
        
    @pipeBase.timeMethod
    def run(self, exposure):

        print "Hello", self.config.widths
        trails = self.finder.getTrails(exposure, self.config.widths)
        return trails
        





class SatelliteConfig(pexConfig.Config):
    narrow = pexConfig.ConfigurableField(target = SatelliteFinderTask,
                                         doc="Search for PSF-width satellite trails")
    broad  = pexConfig.ConfigurableField(target = SatelliteFinderTask,
                                         doc="Search for wide aircraft trails")

class SatelliteRunner(pipeBase.TaskRunner):
    """A custom runner for the SatelliteTask.
    We only need this to pass through kwargs in the getTargetList method.
    """
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs['detectionType'] = parsedCmd.detectionType
        kwargs['debugType'] = parsedCmd.debugType.split(',') if parsedCmd.debugType else ()
        return [(ref, kwargs) for ref in parsedCmd.id.refList]
    
class SatelliteTask(pipeBase.CmdLineTask):
    _DefaultName = "satellite"
    RunnerClass = SatelliteRunner
    ConfigClass = SatelliteConfig

    @classmethod
    def _makeArgumentParser(cls):
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "calexp", help="Data ID, e.g. --id tract=1234 patch=2,2",
                               ContainerClass=pipeBase.DataIdContainer)
        parser.add_argument("-d", "--detectionType",
                            choices=('sat', 'ac', 'all'), default='all', help="Set to run")
        parser.add_argument("-b", '--debugType', default=None,
                            help="Debug outputs to produce (comma-sep-list of plot,trail,fits)")
        return parser

    @classmethod
    def applyOverrides(cls, config):

        # aircraft
        config.broad.luminosityLimit = 0.02
        config.broad.centerLimit     = 1.0      
        config.broad.eRange          = 0.08     
        config.broad.houghBins       = 200      
        config.broad.kernelSigma     = 9             # pixels
        config.broad.kernelWidth     = 15            # pixels
        config.broad.widths          = [40.0, 70.0, 100]  # widths of out of focus aircraft (unbinned)
        config.broad.houghThresh     = 40     
        config.broad.skewLimit       = 50.0
        config.broad.bLimit          = 1.5 
        config.broad.bins            = 8
        config.broad.maxTrailWidth   = 1.6 # a multiple of binning

        # satellites
        config.narrow.luminosityLimit = 0.02  
        config.narrow.centerLimit     = 1.2   
        config.narrow.eRange          = 0.08  
        config.narrow.houghBins       = 200   
        config.narrow.kernelSigma     = 7      # pixels
        config.narrow.kernelWidth     = 11     # pixels
        config.narrow.widths          = [1.0, 8.0]  # widths of satellites and meteors
        config.narrow.houghThresh     = 40    
        config.narrow.skewLimit       = 10.0
        config.narrow.bLimit          = 1.4
        config.narrow.bins            = 4
        config.narrow.maxTrailWidth   = 2.1 # multiple of binning

        
    def __init__(self, *args, **kwargs):
        super(SatelliteTask, self).__init__(*args, **kwargs)
        self.makeSubtask('narrow')
        self.makeSubtask('broad')
        
    
    @pipeBase.timeMethod
    def run(self, dataRef, **kwargs):

        detectionType = kwargs.get('detectionType', 'all')
        debugType     = kwargs.get("debugType", ())

        v,c = dataRef.dataId['visit'], dataRef.dataId['ccd']
        
        self.log.info("Detecting satellite trails in visit=%d, ccd=%d)" % (v,c))
        

        #############################################
        # Debugging: make a place to dump files
        if debugType:
            basedir = os.environ.get('SATELLITE_DATA')
            if not basedir:
                basedir = os.path.join(os.environ.get("PWD"), "data")
            path = os.path.join(basedir, "%04d" %(v))
            try:
                os.mkdir(path)
            except:
                pass

            
        ###############################################
        # Do the work
        ###############################################
            
        exposure = dataRef.get('calexp', immediate=True)
        trails, timing = self.runSatellite(exposure, detectionType=detectionType)

        # mask any trails
        for i, trail in enumerate(trails):
            maskedPixels = trail.setMask(exposure)
            msg = "(%s,%s) Trail %d/%d %s:  maskPix: %d" % (v, c, i+1, len(trails), trail, maskedPixels)
            self.log.info(msg)

        listMsg = "(%s,%s) Detected %d trail(s).  %s" % (v, c, len(trails), trails)
        self.log.info(listMsg)

        
        ################################################
        # Debugging:

        # plot trails
        if 'plot' in debugType:
            def debugPlot(msg, filebase, finder):
                self.log.info("DEBUGGING: Now plotting %s detections." % (msg))
                filename = os.path.join(path,"%s-%05d-%03d.png" % (filebase, v, c))
                satDebug.debugPlot(finder, filename)
            debugPlot("SATELLITE", "satdebug", self.narrow.finder)
            debugPlot("AIRCRAFT",  "acdebug",  self.broad.finder)

        # dump trails to a pickle
        if 'trail' in debugType:
            picfile = os.path.join(path, "trails%05d-%03d.pickle" % (v,c))
            with open(picfile, 'w') as fp:
                bundle = ((v,c), trails, timing)
                pickle.dump(bundle, fp)

        # Debugging: Write to FITS
        if 'fits' in debugType:
            exposure.writeFits(os.path.join(path,"exp%04d-%03d.fits"%(v,c)))

        return trails, timing


    
    def runSatellite(self, exposure, detectionType="all"):
        
        ##############################################
        # Run 2 sweeps ... narrow and broad
        ##############################################

        t0 = time.time()
        trails = satTrail.SatelliteTrailList(0.0, 0.0, 0.0)

        # run for regular satellites
        if detectionType in ('all', 'sat'):
            trailsSat = self.narrow.run(exposure)
            trails = trailsSat.merge(trails, drMax=90.0, dThetaMax=0.15)
        # run for aircraft trails
        if detectionType in ('all', 'ac'):
            trailsAc = self.broad.run(exposure)
            trails = trailsAc.merge(trails, drMax=90.0, dThetaMax=0.15)            
        
        return trails, time.time() - t0

            
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
    
