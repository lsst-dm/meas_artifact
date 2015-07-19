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
                 luminosityMax   = 10.0,
                 skewLimit       = 40.0,
                 bLimit          = 0.5,
                 maxTrailWidth   = 35.0,  # > 30 is rare (even for aircraft)
                 log             = None,
                 verbose         = False,
             ):
        """ """
        
        self.kernelSigma       = kernelSigma        
        self.kernelWidth       = kernelWidth
        self.kx                = np.arange(kernelWidth) - kernelWidth//2
        self.ky                = np.arange(kernelWidth) - kernelWidth//2
        self.bins              = bins
        self.sigmaSmooth       = 1.0

        self.centerLimit       = centerLimit
        self.eRange            = eRange
        self.houghThresh       = houghThresh
        self.houghBins         = houghBins
        self.luminosityLimit   = luminosityLimit
        self.luminosityMax     = luminosityMax
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


    def _makeCalibrationImage(self, psfSigma, width):
        """Make a fake satellite trail with the PSF to calibrate moments we measure.

        @param psfSigma       Gaussian sigma for a double-Gaussian PSF model (in pixels)
        @param width          Width of the trail in pixels (0.0 for PSF alone, but wider for aircraft trails)

        @return calImg        An afwImage containing a fake satellite/aircraft trail
        """

        # tricky.  We have to make a fake trail so it's just like one in the real image

        # To get the binning right, we start with an image 'bins'-times too big
        cx, cy   = (self.bins*self.kernelWidth)//2 - 0.5, 0
        calImg   = afwImage.ImageF(self.bins*self.kernelWidth, self.bins*self.kernelWidth)
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
        rms         = img[isGood].std()
        psfSigma    = satUtil.getExposurePsfSigma(exposure, minor=True)
        #goodDet     = (msk & DET > 0) & isGood
        
        #################################################
        # Faint-trail detection image
        #################################################
        
        # construct a specially clipped image to search for faint trails
        # - zero-out the detected pixels (in addition to otherwise bad pixels)
        exp_faint = exp.clone()
        exp_faint.setMetadata(exposure.getMetadata())
        exp_faint.setPsf(exposure.getPsf())
        msk_faint = exp_faint.getMaskedImage().getMask().getArray()
        img_faint = exp_faint.getMaskedImage().getImage().getArray()
        img_faint[(msk_faint & (MASK | DET) > 0)] = 0.0
        rms_faint = img_faint.std()
        #img_faint[(img_faint < rms_faint)] = rms_faint

        # subtract a small scale background when we search for PSFs
        if np.abs(widths[0]) < 1.1:
            back       = satUtil.medianRing(img_faint, 20.0, 2.0*self.sigmaSmooth)
            img       -= back
            img_faint -= back
        
        #   - smooth 
        img       = satUtil.smooth(img,       self.sigmaSmooth)
        img_faint = satUtil.smooth(img_faint, self.sigmaSmooth)

        
        rms = img_faint[(msk_faint & (MASK | DET) == 0)].std()
        
        #################################################
        # Calibration images
        #################################################
        calImages = []
        for width in widths:
            calImages.append(self._makeCalibrationImage(psfSigma, width))

        
        ################################################
        #
        # Moments
        #
        #################################################
        xx, yy = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))
        
        mm       = momCalc.MomentManager(img, kernelWidth=self.kernelWidth, kernelSigma=self.kernelSigma)
        mm_faint = momCalc.MomentManager(img_faint, kernelWidth=self.kernelWidth, kernelSigma=self.kernelSigma)

        isCandidate = np.zeros(img.shape, dtype=bool)
                
        mmCals = []
        nHits = []
        
        #Selector = momCalc.PixelSelector
        Selector = momCalc.PValuePixelSelector
        maxPixels = 1000
        for i, calImg in enumerate(calImages):
            mmCal = momCalc.MomentManager(calImg, kernelWidth=self.kernelWidth, kernelSigma=self.kernelSigma, 
                                          isCalibration=True)
            mmCals.append(mmCal)

            maxFactors   = 10.0, 20.0, 500.0
            luminFactors = self.luminosityLimit*rms,  10.0,  20.0
            scaleFactors = 1.0,   2.0,   3.0
            
            sumI  = momCalc.MomentLimit('sumI',        self.luminosityLimit*rms, 'lower')  #dummy value
            lumX  = momCalc.MomentLimit('sumI',        self.luminosityLimit*rms, 'upper')  #dummy value
            cent  = momCalc.MomentLimit('center',      2.0*self.centerLimit,     'center')
            centP = momCalc.MomentLimit('center_perp', self.centerLimit,         'center')
            skew  = momCalc.MomentLimit('skew',        2.0*self.skewLimit,       'center')
            skewP = momCalc.MomentLimit('skew_perp',   self.skewLimit,           'center')
            ellip = momCalc.MomentLimit('ellip',       self.eRange,              'center')
            b     = momCalc.MomentLimit('b',           self.bLimit,              'center')

            selector = Selector(mm, mmCal)
            for limit in ellip, sumI, cent, centP, skew, skewP, b: #, lumX:
                selector.append(limit)

            isCand = np.zeros(img.shape, dtype=bool)
            pixelSums = []
            for maxFact, luminFact, scaleFact in zip(maxFactors, luminFactors, scaleFactors):
                sumI.norm   = luminFact
                lumX.norm   = maxFact
                cent.norm  /= scaleFact
                centP.norm /= scaleFact
                skew.norm  /= scaleFact
                skewP.norm /= scaleFact
                ellip.norm *= scaleFact
                b.norm     *= scaleFact
                
                pixels      = selector.getPixels(maxPixels=maxPixels)
                isCand |= pixels
                pixelSums.append(pixels.sum())

                
            if True:
                selector = Selector(mm_faint, mmCal)
                sumI    = momCalc.MomentLimit('sumI',        3.0*rms,                  'lower')
                lumX    = momCalc.MomentLimit('sumI',        5.0*rms,                  'upper')
                cent    = momCalc.MomentLimit('center',      2.0*self.centerLimit,     'center')
                skew    = momCalc.MomentLimit('skew',        2.0*self.skewLimit,       'center')
                ellipLo = momCalc.MomentLimit('ellip',       0.10,                     'lower')
                ellipHi = momCalc.MomentLimit('ellip',       0.90,                     'upper')
                for limit in sumI, cent, skew, ellipHi, ellipLo:
                    selector.append(limit)
                    
                faintPixels  = selector.getPixels(maxPixels=maxPixels) & (msk & (MASK | DET)==0)
                isCand |= faintPixels
                pixelSums.append(faintPixels.sum())
            else:
                pixelSums.append(0)
                
            isCandidate |= isCand
            
            msg = "Candidates: nPix/med/bri = %d/ %d/ %d   faint: %d  totals: %d/ %d" % (
                pixelSums[0], pixelSums[1], pixelSums[2], pixelSums[3],
                isCand.sum(), isCandidate.sum()
            )
            self.log.logdebug(msg)
                
            nHits.append((widths[i], isCand.sum()))
        
        bestCal = sorted(nHits, key=lambda x: x[1], reverse=True)[0]
        bestWidth = bestCal[0]

        nBeforeAlignment = isCandidate.sum()
        thetaMatch, newTheta = hough.thetaAlignment(mm.theta[isCandidate],xx[isCandidate],yy[isCandidate],
                                                    limit=5)

        mm.theta[isCandidate] = newTheta
        isCandidate[isCandidate] = thetaMatch
        nAfterAlignment = isCandidate.sum()
        self.log.logdebug("theta-alignment Bef/aft: %d / %d" % (nBeforeAlignment, nAfterAlignment))

        #################################################
        # Hough transform
        #################################################
        rMax           = sum([q**2 for q in img.shape])**0.5
        houghTransform = hough.HoughTransform(self.houghBins, self.houghThresh,
                                              rMax=rMax, maxPoints=1000, nIter=0, maxResid=4.5)
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
                trails.append(trail)

        self._mm           = mm
        self._mmCals       = mmCals
        self._isCandidate  = isCandidate
        self._brightFactor = 10
        self._trails       = trails
        self._solutions    = solutions
        
        return trails

        
