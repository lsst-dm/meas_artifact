#!/usr/bin/env python

import os
import time
import numpy as np

import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

import lsst.pex.logging as pexLog
import lsst.afw.image   as afwImage
import lsst.afw.math    as afwMath

import hough
import satelliteUtils   as satUtil
import satelliteTrail   as satTrail
import momentCalculator as momCalc
import satelliteDebug   as satDebug


class SatelliteFinder(object):

    def __init__(self,
                 kernelSigma     = 15,
                 kernelWidth     = 31,
                 bins            = 4,
                 centerLimit     = 0.8,
                 eRange          = 0.06,
                 houghThresh     = 20,
                 houghBins       = 256,
                 luminosityLimit = 4.0,
                 skewLimit       = 40.0,
                 bLimit          = 0.5,
                 maxTrailWidth   = 5.5,
                 log             = None,
                 verbose         = False,
             ):
        """ """
        
        self.kernelSigma       = kernelSigma        
        self.kernelWidth       = kernelWidth
        self.bins              = bins
        self.sigmaSmooth       = 1.0

        self.centerLimit       = centerLimit
        self.eRange            = eRange
        self.houghThresh       = houghThresh
        self.houghBins         = houghBins
        self.luminosityLimit   = luminosityLimit
        self.skewLimit         = skewLimit
        self.bLimit            = bLimit

        self.maxTrailWidth     = maxTrailWidth
        
        if log is None:
            logLevel = pexLog.Log.INFO
            if verbose:
                logLevel = pexLog.Log.DEBUG
            log = pexLog.Log(pexLog.Log.getDefaultLog(), 'satelliteFinder', logLevel)
        self.log = log
        
        self.debugInfo = {}


    def _makeCalibrationImage(self, psfSigma, width, kernelWidth=None, kernelSigma=None):
        """Make a fake satellite trail with the PSF to calibrate moments we measure.

        @param psfSigma       Gaussian sigma for a double-Gaussian PSF model (in pixels)
        @param width          Width of the trail in pixels (0.0 for PSF alone, but wider for aircraft trails)
        @param kernelWidth
        @param kernelSigma
        
        @return calImg        An afwImage containing a fake satellite/aircraft trail
        """

        kernelWidth = kernelWidth or self.kernelWidth
        kernelSigma = kernelSigma or self.kernelSigma
        
        # tricky.  We have to make a fake trail so it's just like one in the real image

        # To get the binning right, we start with an image 'bins'-times too big
        cx, cy   = (self.bins*kernelWidth)//2 - 0.5, 0
        calImg   = afwImage.ImageF(self.bins*kernelWidth, self.bins*kernelWidth)
        calArr   = calImg.getArray()

        # Make a trail with the requested (unbinned) width
        calTrail = satTrail.SatelliteTrail(cx, cy)

        # for wide trails, just add a constant with the stated width
        if width > 8.0*psfSigma:
            profile = satTrail.ConstantProfile(1.0, width)
            insertWidth = width
        # otherwise, use a double gaussian
        else:
            profile  = satTrail.DoubleGaussianProfile(1.0, width/2.0 + psfSigma)
            insertWidth = 4.0*(width/2.0 + psfSigma)
        calTrail.insert(calArr, profile, insertWidth)

        if False:
            wDet = calArr > 0.0002
            calArr[wDet] = 1.0
            calArr[~wDet] = 0.0
        # Now bin and smooth, just as we did the real image
        calArr   = afwMath.binImage(calImg, self.bins).getArray()
        calArr   = satUtil.smooth(calArr, self.sigmaSmooth)

        if False:
            fig = figure.Figure()
            can = FigCanvas(fig)
            ax = fig.add_subplot(111)
            ax.imshow(calArr, interpolation='none', cmap='gray')
            fig.savefig("satellite-calib-image.png")

        return calArr

        
    def getTrails(self, exposure, widths):

        emsk = exposure.getMaskedImage().getMask()
        DET  = emsk.getPlaneBitMask("DETECTED")
        MASK = 0
        for plane in "BAD", "CR", "SAT", "INTRP", "EDGE", "SUSPECT":
            MASK |= emsk.getPlaneBitMask(plane)
        
        t1 = time.time()

        #################################################
        # Main detection image
        #################################################
        
        if self.bins == 1:
            exp = exposure.clone()
        else:
            exp = type(exposure)(afwMath.binImage(exposure.getMaskedImage(), self.bins))
            exp.setMetadata(exposure.getMetadata())
            exp.setPsf(exposure.getPsf())

        img         = exp.getMaskedImage().getImage().getArray()
        msk         = exp.getMaskedImage().getMask().getArray()
        isBad       = msk & MASK > 0
        isGood      = ~isBad
        img[isBad]  = 0.0         # convolution will smear bad pixels.  Zero them out.
        psfSigma    = satUtil.getExposurePsfSigma(exposure, minor=True)

        
        #################################################
        # Faint-trail detection image
        #################################################
        
        # construct a specially clipped image to use to model a fine scale background
        # - zero-out the detected pixels (in addition to otherwise bad pixels)
        expClip = exp.clone()
        mskClip = expClip.getMaskedImage().getMask().getArray()
        imgClip = expClip.getMaskedImage().getImage().getArray()
        imgClip[(mskClip & (MASK | DET) > 0)] = 0.0

        # subtract a small scale background when we search for PSFs
        if np.abs(widths[0]) < 1.1:
            self.sigmaSmooth = self.sigmaSmooth
            back       = satUtil.medianRing(imgClip, self.kernelWidth, 2.0*self.sigmaSmooth)
            wDet       = msk & DET > 0
            sig = imgClip.std()
            wSig = img > 2.0*sig
            # amplify detected pixels (make this configurable?)
            img[wDet|wSig] *= 10.0
            img       -= back
            imgClip   -= back
            kernelGrow = 1.4
            thetaTol   = 0.15
        else:
            self.sigmaSmooth = 2.0
            kernelGrow = 1.4
            thetaTol   = 0.25
        
        #   - smooth 
        img       = satUtil.smooth(img,       self.sigmaSmooth)
        imgClip   = satUtil.smooth(imgClip, self.sigmaSmooth)
        rms       = imgClip[(mskClip & (MASK | DET) == 0)].std()

        
        isCandidate = np.ones(img.shape, dtype=bool)

        ###########################################
        # Try different kernel sizes
        ###########################################
        
        # Different sized kernels should give the same results for a real trail
        # but would be less likely to for noise.
        # Unfortunately, this is costly, and the effect is small.
        for kernelFactor in (1.0, kernelGrow):
            kernelWidth = 2*int((kernelFactor*self.kernelWidth)//2) + 1
            kernelSigma = kernelFactor*self.kernelSigma 

            #print "Kernels:", kernelWidth, kernelSigma
            isKernelCandidate = np.zeros(img.shape, dtype=bool)
        
            
            #################################################
            # Calibration images
            #################################################
            calImages = []
            for width in widths:
                calImages.append(self._makeCalibrationImage(psfSigma, width,
                                                            kernelWidth=kernelWidth, kernelSigma=kernelSigma))


            ################################################
            # Moments
            #################################################
            mm       = momCalc.MomentManager(img, kernelWidth=kernelWidth, kernelSigma=kernelSigma)

            mmCals = []
            nHits = []

            #Selector = momCalc.PixelSelector
            Selector = momCalc.PValuePixelSelector
            maxPixels = 4000
            for i, calImg in enumerate(calImages):
                mmCal = momCalc.MomentManager(calImg, kernelWidth=kernelWidth, kernelSigma=kernelSigma, 
                                              isCalibration=True)
                mmCals.append(mmCal)

                sumI  = momCalc.MomentLimit('sumI',        self.luminosityLimit*rms, 'lower')
                cent  = momCalc.MomentLimit('center',      2.0*self.centerLimit,     'center')
                centP = momCalc.MomentLimit('center_perp', self.centerLimit,         'center')
                skew  = momCalc.MomentLimit('skew',        2.0*self.skewLimit,       'center')
                skewP = momCalc.MomentLimit('skew_perp',   self.skewLimit,           'center')
                ellip = momCalc.MomentLimit('ellip',       self.eRange,              'center')
                b     = momCalc.MomentLimit('b',           self.bLimit,              'center')

                selector = Selector(mm, mmCal)
                for limit in sumI, ellip, cent, centP, skew, skewP, b:
                    selector.append(limit)

                pixels      = selector.getPixels(maxPixels=maxPixels)
                    
                isKernelCandidate |= pixels

                msg = "cand: nPix: %d  tot: %d" % (pixels.sum(), isKernelCandidate.sum())
                self.log.logdebug(msg)

            isCandidate &= isKernelCandidate
            self.log.logdebug("total: %d" % (isCandidate.sum()))

            nHits.append((widths[i], isKernelCandidate.sum()))
        
        bestCal = sorted(nHits, key=lambda x: x[1], reverse=True)[0]
        bestWidth = bestCal[0]

        ###############################################
        # Theta Alignment
        ###############################################
        xx, yy = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))
        nBeforeAlignment = isCandidate.sum()
        maxSeparation = min([x/2 for x in img.shape])
        thetaMatch, newTheta = hough.thetaAlignment(mm.theta[isCandidate],xx[isCandidate],yy[isCandidate],
                                                    tolerance=thetaTol,limit=3,maxSeparation=maxSeparation)

        mm.theta[isCandidate] = newTheta
        isCandidate[isCandidate] = thetaMatch
        nAfterAlignment = isCandidate.sum()
        self.log.logdebug("theta-alignment Bef/aft: %d / %d" % (nBeforeAlignment, nAfterAlignment))

        #################################################
        # Hough transform
        #################################################
        rMax           = np.linalg.norm(img.shape)
        houghTransform = hough.HoughTransform(self.houghBins, self.houghThresh,
                                              rMax=rMax, maxPoints=1000, nIter=1, maxResid=5.5)
        solutions      = houghTransform(mm.theta[isCandidate], xx[isCandidate], yy[isCandidate])

        #################################################
        # Trail objects
        #################################################
        trails = satTrail.SatelliteTrailList(nAfterAlignment, solutions.binMax, psfSigma)
        for s in solutions:
            trail = satTrail.SatelliteTrail.fromHoughSolution(s, self.bins)
            trail.measure(exp, bins=self.bins)
            # last chance to drop it
            if trail.width < self.maxTrailWidth:
                self.log.info(str(trail))
                trails.append(trail)
            else:
                self.log.info("Dropping (maxWidth>%.1f): %s" %(self.maxTrailWidth, trail))
                
        self._mm           = mm
        self._mmCals       = mmCals
        self._isCandidate  = isCandidate
        self._brightFactor = 10
        self._trails       = trails
        self._solutions    = solutions
        
        return trails

        
