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


#########################################################################################
#
# SatelliteFinderTask is a simple Task wrapper for the SatelliteFinder class
#
#########################################################################################

class SatelliteFinderConfig(pexConfig.Config):
    bins            = pexConfig.Field(dtype=int,   default=4,    doc="How to bin the image before detection")
    doBackground    = pexConfig.Field(dtype=bool,  default=True, doc="Median-ring-filter background subtract")
    scaleDetected   = pexConfig.Field(dtype=float, default=10.0, doc="Scale detected pixels by this amount.")
    sigmaSmooth     = pexConfig.Field(dtype=float, default=1.0,  doc="Gaussian smooth sigma (binned pixels)")
    thetaTolerance  = pexConfig.Field(dtype=float, default=0.15,
                                      doc="Max theta difference for thetaAlignment() routine.")
    luminosityLimit = pexConfig.Field(dtype=float, default=0.02, doc="Lowest luminosity in Std.Dev.")
    centerLimit     = pexConfig.Field(dtype=float, default=1.2,  doc="Max centroid value [pixels]")
    eRange          = pexConfig.Field(dtype=float, default=0.08, doc="Max ellipticity range.")
    bLimit          = pexConfig.Field(dtype=float, default=1.4,  doc="Max error in x-cor trail width.")
    skewLimit       = pexConfig.Field(dtype=float, default=10.0, doc="Max value of x-cor skew (3rd moment)")
    
    kernelSigma     = pexConfig.Field(dtype=float, default=7.0,  doc="Gauss sigma to use for x-cor kernel.")
    kernelWidth     = pexConfig.Field(dtype=int,   default=11,   doc="Width of x-cor kernel in pixels")
    growKernel      = pexConfig.Field(dtype=float, default=1.4,
                                      doc="Repeat with a kernel larger by this fraction (no repeat if 1.0)")
    
    houghBins       = pexConfig.Field(dtype=int,   default=200,  doc="Num. of bins to use in r,theta space.")
    houghThresh     = pexConfig.Field(dtype=int,   default=40,
                                      doc="Min number of 'hits' in a Hough bin for a detection.")
    maxTrailWidth   = pexConfig.Field(dtype=float, default=2.1,
                                      doc="Discard trails with measured widths greater than this (pixels).")
    
    maskAndBits     = pexConfig.ListField(dtype=str,   default=(),
                                      doc="Only mask pixels with one of these bits set.")
    widths          = pexConfig.ListField(dtype=float, default= (1.0, 8.0),
                                          doc="*unbinned* width of trail to search for.")

class SatelliteFinderTask(pipeBase.Task):
    """A thin Task wrapper to construct and run a SatelliteFinder."""
    
    _DefaultName = 'satelliteFind'
    ConfigClass = SatelliteFinderConfig


    def __init__(self, *args, **kwargs):
        """Constructor. Instantiate a SatelliteFinder object as an attribute."""
        
        super(SatelliteFinderTask,self).__init__(*args, **kwargs)
        self.finder = satFind.SatelliteFinder(
            bins            = self.config.bins,
            doBackground    = self.config.doBackground,
            scaleDetected   = self.config.scaleDetected,
            sigmaSmooth     = self.config.sigmaSmooth,
            thetaTolerance  = self.config.thetaTolerance,
            
            luminosityLimit = self.config.luminosityLimit,
            centerLimit     = self.config.centerLimit,
            eRange          = self.config.eRange,
            bLimit          = self.config.bLimit,
            skewLimit       = self.config.skewLimit,
            
            kernelSigma     = self.config.kernelSigma,
            kernelWidth     = self.config.kernelWidth,
            growKernel      = self.config.growKernel,
            
            houghThresh     = self.config.houghThresh,
            houghBins       = self.config.houghBins,
            maxTrailWidth   = self.config.maxTrailWidth*self.config.bins, # correct for binning
            maskAndBits     = self.config.maskAndBits,
            log             = self.log
        )
        
    @pipeBase.timeMethod
    def run(self, exposure):
        """Run detection with the SatelliteFinder, and return detected trails.
        """
        trails = self.finder.getTrails(exposure, self.config.widths)
        return trails



#########################################################################################
#
# SatelliteTask is a base class for all CmdLineTask Satellite codes.
#
# It should be straightforward (example below) to use a different satellite detection code
# by:
#    - wrapping it in a task
#    - writing a derived Task which inherits from this SatelliteTask and calls
#      your wrapped task in overloaded runSatellite() method.
#
#########################################################################################
class SatelliteConfig(pexConfig.Config):    
    debugType        = pexConfig.Field(dtype=str,   default="",
                                       doc="Types debug output to write (fits,trail)")

    defaultDir = os.path.join(os.environ.get("PWD"), "data")
    debugDir         = pexConfig.Field(dtype=str, default=defaultDir,
                                       doc="Directory to write debug outputs")
    
    doMask     = pexConfig.Field(dtype=bool, default=True, doc="Mask detected satellite trails?")
    
class SatelliteTask(pipeBase.CmdLineTask):
    """Detect and mask Satellite trails and other linear features.
    """

    _DefaultName = "satellite"
    ConfigClass = SatelliteConfig

    @classmethod
    def _makeArgumentParser(cls):
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "calexp", help="Data ID, e.g. --id tract=1234 patch=2,2",
                               ContainerClass=pipeBase.DataIdContainer)
        return parser

    def __init__(self, *args, **kwargs):
        """Construct
        """
        # We don't use schema, but we may some day, and keeping it is
        # consistent with other pipeline tasks.
        
        schema = kwargs.pop("schema", None)
        super(SatelliteTask, self).__init__(*args, **kwargs)
        
    
    @pipeBase.timeMethod
    def run(self, dataRef, **kwargs):
        """Called when run as a CmdLineTask (use process() method otherwise).

        This is a simple wrapper for our own runSatellite() method, but includes
        extra debugging utilities to plot and store other relevant debug information.

        @param dataRef              Butler dataRef object specifying e.g. visit,ccd to run.
        @param kwargs['debugType']  Specify debug data to output (options 'trail', 'fits')
        
        @return trails,timing        
        -- trails               SatelliteTrailList object containing detections
        -- timing               Runtime in seconds (for debugging).

        The keyword arg 'debugType' is a list provided on the command line to specify
        any debugging outputs.  The base class allows 'trail' (pickle the trails)
        and 'fits' (write the FITS image with trails masked).
        """

        # these should be replaced by 'visit-like-thing' and 'ccd-like-thing'
        # The values are used only in logs and in filenames used when
        # debugging is enabled.
        v,c = dataRef.dataId['visit'], dataRef.dataId['ccd']
                
        ###############################################
        # Do the work
        ###############################################
        self.log.info("Detecting satellite trails in visit=%d, ccd=%d)" % (v,c))
        
        exposure = dataRef.get('calexp', immediate=True)
        trails, timing = self.process(exposure)

        
        ################################################
        # Debugging:
        #
        # Debugging may grind your process to a halt
        # as some of these things produce plots and are
        # very slow. 
        ################################################
        debugDir = self.config.debugDir
        path = os.path.join(debugDir, "%04d" %(v))
        debugType     = self.config.debugType
        debugType = debugType.split(",") if debugType else ()
        if debugType:
            try:
                os.mkdir(path)
            except:
                pass

        # dump trails to a pickle
        if 'trail' in debugType:
            picfile = os.path.join(path, "trails%05d-%03d.pickle" % (v,c))
            self.log.info("DEBUGGING: Pickling results in %s." % (picfile))
            with open(picfile, 'w') as fp:
                bundle = ((v,c), trails, timing)
                pickle.dump(bundle, fp)

        # Debugging: Write to FITS
        if 'fits' in debugType:
            fitsfile = os.path.join(path,"exp%04d-%03d.fits"%(v,c))
            self.log.info("DEBUGGING: Writing FITS in %s." % (fitsfile))
            exposure.writeFits(fitsfile)
        
        self.runDebug(dataRef, path, debugType)

        return trails, timing

    
    def process(self, exposure):
        """Run satellite detection and mask any detected trails.

        @param exposure     The exposure to process.
        
        @return trails,timing        
        -- trails               SatelliteTrailList object containing detections
        -- timing               Runtime in seconds (for debugging).

        This method is intended to be the one-stop entry point for pipeline
        calls.  It accepts an exposure directly, and run detection and masking,
        and return the trails and timing.
        """

        trails, timing = self.runSatellite(exposure)
        msg = "Detected %d trail(s) in %.2f sec." % (len(trails), timing)
        self.log.info(msg)
        nMaskedPixels = self.setMask(exposure, trails)
        return trails, timing

    
    def setMask(self, exposure, trails):
        """Set the mask plane for trails in exposure.

        @param exposure        Exposure containing the detected satellite trails.
        @param trails          The SateliteTrailList object containing trails.

        Derived classes shouldn't need to overload this method. 
        """

        if not self.config.doMask:
            return 0
        
        ###################
        # mask any trails
        msk            = exposure.getMaskedImage().getMask()
        satellitePlane = msk.addMaskPlane("SATELLITE")
        satelliteBit   = msk.getPlaneBitMask("SATELLITE")
        nMaskedPixels = 0
        for i, trail in enumerate(trails):
            maskedPixels = trail.setMask(exposure, satelliteBit=satelliteBit)
            msg = "  Trail %d/%d %s:  maskPix: %d" % (i+1, len(trails), trail, maskedPixels)
            self.log.info(msg)
            nMaskedPixels += maskedPixels
        return nMaskedPixels

    def runSatellite(self, exposure):
        """Method to run satellite detection routine.

        @param exposure   The exposure to run.

        @return trails,timing
        -- trails               SatelliteTrailList object containing detections
        -- timing               Runtime in seconds (for debugging).

        THIS MUST BE OVERLOADED IN DERIVED CLASS
        """
        raise NotImplementedError("You must implement runSatellite() in your derived class.")

    def runDebug(self, dataRef, path, debugType):
        """Method to run custom debugging routine.

        @param dataRef        Butler DataRef for e.g. visit,ccd
        @param path           Location to write any debug output files.
        
        THIS MUST BE OVERLOADED IN DERIVED CLASS
        """
        raise NotImplementedError("You must implement runDebug() in your derived class.")

    
    # This Task should have no reason to touch the data repo
    # Disable anything that might try to do so.
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None



###############################################################
#
# Concrete implementation of the Hough SatelliteTask
#
# We'll inherit from SatelliteTask and overload methods
# runSatellite()
# runDebut()
################################################################

    
class HoughSatelliteConfig(SatelliteConfig):

    narrow   = pexConfig.ConfigurableField(target = SatelliteFinderTask,
                                         doc="Search for PSF-width satellite trails")
    broad    = pexConfig.ConfigurableField(target = SatelliteFinderTask,
                                         doc="Search for wide aircraft trails")

    doNarrow = pexConfig.Field(dtype=bool, default=True, doc="Run narrow trail detection.")
    doBroad  = pexConfig.Field(dtype=bool, default=True, doc="Run broad trail detection.")

    def setDefaults(self):
        # satellites
        self.narrow.bins            = 4
        self.narrow.doBackground    = True
        self.narrow.scaleDetected   = 10.0
        self.narrow.sigmaSmooth     = 1.0
        self.narrow.thetaTolerance  = 0.15
        self.narrow.widths          = [1.0, 8.0]  # widths of satellites and meteors
        self.narrow.luminosityLimit = 0.02
        self.narrow.centerLimit     = 1.2
        self.narrow.eRange          = 0.08
        self.narrow.bLimit          = 1.4
        self.narrow.skewLimit       = 10.0
        self.narrow.kernelSigma     = 7      # pixels
        self.narrow.kernelWidth     = 11     # pixels
        self.narrow.growKernel      = 1.4
        self.narrow.houghBins       = 200
        self.narrow.houghThresh     = 25
        self.narrow.maxTrailWidth   = 2.0 # multiple of binning
        self.narrow.maskAndBits     = ("DETECTED",)

        # out-of-focus aircraft default
        self.broad.bins            = 8
        self.broad.doBackground    = False
        self.broad.scaleDetected   = 1.0
        self.broad.sigmaSmooth     = 2.0
        self.broad.thetaTolerance  = 0.25
        self.broad.widths          = [40.0, 70.0, 100]  # widths of out of focus aircraft (unbinned)
        self.broad.luminosityLimit = 0.02
        self.broad.centerLimit     = 1.0      
        self.broad.eRange          = 0.08     
        self.broad.bLimit          = 1.5
        self.broad.skewLimit       = 50.0
        self.broad.kernelSigma     = 9             # pixels
        self.broad.kernelWidth     = 15            # pixels
        self.broad.growKernel      = 1.4 
        self.broad.houghBins       = 200      
        self.broad.houghThresh     = 50     
        self.broad.maxTrailWidth   = 2.0 # a multiple of binning
        self.broad.maskAndBits     = ()


        
class HoughSatelliteTask(SatelliteTask):
    """Detect and mask Satellite trails and other linear features.
    """

    _DefaultName = "houghSatellite"
    ConfigClass = HoughSatelliteConfig


    def __init__(self, *args, **kwargs):
        super(SatelliteTask, self).__init__(*args, **kwargs)
        self.makeSubtask('narrow')
        self.makeSubtask('broad')
        
    
    def runSatellite(self, exposure):
        """Run detection for both narrow and broad satellite trails.

        @param exposure   Calibrated exposure to run detection on.
        
        @return trails,timing
        -- trails               SatelliteTrailList object containing detections
        -- timing               Runtime in seconds (for debugging).
        """
        
        ##############################################
        # Run 2 sweeps ... narrow and broad
        ##############################################

        t0 = time.time()
        trails = satTrail.SatelliteTrailList(0.0, 0.0, 0.0)

        # run for regular satellites
        if self.config.doNarrow:
            trailsSat = self.narrow.run(exposure)
            trails = trailsSat.merge(trails, drMax=90.0, dThetaMax=0.15)
        # run for aircraft trails
        if self.config.doBroad:
            trailsAc = self.broad.run(exposure)
            trails = trailsAc.merge(trails, drMax=90.0, dThetaMax=0.15)            
        
        return trails, time.time() - t0


    def runDebug(self, dataRef, path, debugType):
        """Custom debug output for this SatelliteFinder
        """

        v,c = dataRef.dataId['visit'], dataRef.dataId['ccd']
        
        # plot trails
        if 'plot' in debugType:
            def debugPlot(msg, filebase, finder):
                self.log.info("DEBUGGING: Now plotting %s detections." % (msg))
                filename = os.path.join(path,"%s-%05d-%03d.png" % (filebase, v, c))
                satDebug.debugPlot(finder, filename)
            if self.config.doNarrow:
                debugPlot("SATELLITE", "satdebug", self.narrow.finder)
            if self.config.doBroad:
                debugPlot("AIRCRAFT",  "acdebug",  self.broad.finder)

            




##########################################################################
#
# Example of how to add another Sat task
#
# As long as your Task uses the same SatelliteTrail objects,
# and is wrapped in a Task,
# you should be able to create your own Config, and
# overload runSatellite() and runDebug() methods
###########################################################################
    

class AnotherSatelliteConfig(pexConfig.Config):
    dummy = pexConfig.ConfigurableField(target = SatelliteFinderTask,
                                         doc="Search for PSF-width satellite trails")
    doDummy = pexConfig.Field(dtype=bool, default=True, doc="")

    def setDefaults(self):
        # satellites
        self.dummy.widths          = [1.0,],
        self.dummy.houghThresh     = 40


class AnotherSatelliteTask(SatelliteTask):
    """Detect and mask Satellite trails and other linear features ... a different way.
    """

    _DefaultName = "anotherSatellite"
    ConfigClass = AnotherSatelliteConfig

    def __init__(self, *args, **kwargs):
        super(SatelliteTask, self).__init__(*args, **kwargs)
        self.makeSubtask('dummy')

        
    def runSatellite(self, exposure):
        """Run detection 

        @param exposure   Calibrated exposure to run detection on.
        
        @return trails,timing
        -- trails               SatelliteTrailList object containing detections
        -- timing               Runtime in seconds (for debugging).
        """
        
        t0 = time.time()
        trails = satTrail.SatelliteTrailList(0.0, 0.0, 0.0)

        # run for regular satellites
        if self.config.doDummy:
            trails = self.dummy.run(exposure)
            
        return trails, time.time() - t0


    def runDebug(self, dataRef, **kwargs):
        """Custom debug output for AnotherSatelliteFinder
        """
        debugType     = kwargs.get("debugType", ())
        v,c = dataRef.dataId['visit'], dataRef.dataId['ccd']
        
        # plot trails
        if 'plot' in debugType:
            def debugPlot(msg, filebase, finder):
                self.log.info("DEBUGGING: Now plotting %s detections." % (msg))
                filename = os.path.join(path,"%s-%05d-%03d.png" % (filebase, v, c))
                satDebug.debugPlot(finder, filename)
            if self.config.doDummy:
                debugPlot("SATELLITE", "satdebug", self.dummy.finder)





##########################################################################
#
# Pool processing
#
###########################################################################
    
                   
class PoolSatelliteConfig(pexConfig.Config):
    satellite    = pexConfig.ConfigurableField(target=HoughSatelliteTask, doc="satellite")

    
class PoolSatelliteTask(basePara.BatchPoolTask):
    """A batch job processor for SatelliteTask

    You should almost never need to run this unless you're doing some development
    of the SatelliteFinder algorithm itself.  But if you do need it, it works
    just like e.g. reduceFrames.py or multiBand.py (i.e. the pipeline batch processors)
    """
    
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
        parser.add_id_argument("--id", "calexp", level="visit", ContainerClass=pipeBase.DataIdContainer,
                               help="data ID, e.g. --id visit=12345")
        return parser

    @basePool.abortOnError
    def run(self, expRef, butler):
        """Handle the scatter/gather (or map/reduce if you prefer) for satellite detection jobs.

        There's no real 'gather' her per-se ... just a scatter.  The results are returned directly.

        @param expRef   The exposure reference to run.
        @param butler   The butler to get the data.
        """
        
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
        """Process this dataId."""
        
        dataRef = hscButler.getDataRef(cache.butler, dataId, datasetType="src")
        ccdId = dataRef.get("ccdExposureId")

        with self.logOperation("Started satellite %s (ccdId=%d) on %s" % (dataId, ccdId, basePool.NODE)):
            try:
                # *** HERE's the important call ***
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

