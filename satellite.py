#!/usr/bin/env python

import os
import time
import numpy as np

import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

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
             ):
        """ """
        
        self.kernelSigma       = kernelSigma        
        self.kernelWidth       = kernelWidth
        self.kx = np.arange(kernelWidth) - kernelWidth//2
        self.ky = np.arange(kernelWidth) - kernelWidth//2
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
        calTrail = satTrail.SatelliteTrail(cx, cy, width=width)
        maskBit  = 1.0 if width > 1.0 else None
        calTrail.insert(calArr, sigma=psfSigma, maskBit=maskBit)

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

        img           = exp.getMaskedImage().getImage().getArray()
        msk           = exp.getMaskedImage().getMask().getArray()
        whereBad      = msk & MASK > 0
        whereGood     = ~whereBad
        img[whereBad] = 0.0         # convolution will smear bad pixels.  Zero them out.
        rms           = img[whereGood].std()
        psfSigma      = satUtil.getExposurePsfSigma(exposure, minor=True)

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

        #   - smooth 
        img       = satUtil.smooth(img,       self.sigmaSmooth)
        img_faint = satUtil.smooth(img_faint, self.sigmaSmooth)
        
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
        
        mm       = momCalc.MomentManager(img, kernelWidth=self.kernelWidth, kernelSigma=self.kernelSigma)
        mm_faint = momCalc.MomentManager(img_faint, kernelWidth=self.kernelWidth, kernelSigma=self.kernelSigma)

        isCandidate = np.zeros(img.shape, dtype=bool)
                
        mmCals = []
        nHits = []

        Selector = momCalc.PixelSelector
        #Selector = momCalc.PValuePixelSelector

        for i, calImg in enumerate(calImages):
            mmCal = momCalc.MomentManager(calImg, kernelWidth=self.kernelWidth, kernelSigma=self.kernelSigma, 
                                          isCalibration=True)
            mmCals.append(mmCal)

            limits = {
                'sumI'         : -self.luminosityLimit*rms,
                'center'       : 2.0*self.centerLimit,
                'center_perp'  : self.centerLimit,
                'skew'         : 2.0*self.skewLimit,
                'skew_perp'    : self.skewLimit,
                'ellip'        : self.eRange,
                'b'            : self.bLimit,
                }

            mediumFactor = 2.0
            mediumLimit   = 10.0*self.luminosityLimit
            mediumLimits = {
                'sumI'         : -mediumLimit*rms,
                'center'       : 2.0*self.centerLimit/mediumFactor,
                'center_perp'  : self.centerLimit/mediumFactor,
                'skew'         : 2.0*self.skewLimit/  mediumFactor,
                'skew_perp'    : self.skewLimit/  mediumFactor,
                'ellip'        : mediumFactor*self.eRange,
                'b'            : mediumFactor*self.bLimit,
                }
            
            brightFactor = 6.0
            brightLimit = 20.0*self.luminosityLimit
            brightLimits = {
                'sumI'         : -brightLimit*rms,
                'center'       : 2.0*self.centerLimit/brightFactor,
                'center_perp'  : self.centerLimit/brightFactor,
                'skew'         : 2.0*self.skewLimit/  brightFactor,
                'skew_perp'    : self.skewLimit/  brightFactor,
                'ellip'        : brightFactor*self.eRange,
                'b'            : brightFactor*self.bLimit,
                }

            pixels = np.zeros(img.shape, dtype=bool)
            for lim in limits, mediumLimits, brightLimits:
                selector = Selector(mm, mmCal, lim)
                pixels |= selector.getPixels()


            # faint trails
            faint_limits = {
                'sumI'         : -2.5*rms,
                #'center'       : 8.0*self.centerLimit,
                #'center_perp'  : 4.0*self.centerLimit,
                #'skew'         : 8.0*self.skewLimit,
                #'skew_per'     : 4.0*self.skewLimit,
                #'ellip'        : 8.0*self.eRange,
                #'b'            : 2.0,
            }                
            selector = Selector(mm_faint, mmCal, faint_limits)
            faint_pixels = selector.getPixels() & (msk & (MASK | DET) == 0)
            
            if pixels.sum() < 150:
                pixels |= faint_pixels

            nHits.append((widths[i], pixels.sum()))

            isCandidate |= pixels

        isCandidate &= ~whereBad
        bestCal = sorted(nHits, key=lambda x: x[1], reverse=True)[0]
        bestWidth = bestCal[0]

        nCandidatePixels = isCandidate.sum()

        ################################################
        # Hough transform
        #################################################
        xx, yy = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))

        rMax           = sum([q**2 for q in img.shape])**0.5
        houghTransform = hough.HoughTransform(self.bins, self.houghThresh, rMax=rMax, maxPoints=1000, nIter=1)
        solutions      = houghTransform(mm.theta[isCandidate], xx[isCandidate], yy[isCandidate])

        #################################################
        # Trail objects
        #################################################
        trails = satTrail.SatelliteTrailList(nCandidatePixels, solutions.binMax, psfSigma)
        for s in solutions:
            print "Trail: ", self.bins*s.r, s.theta, bestWidth, s.binMax
            trail = satTrail.SatelliteTrail(self.bins*s.r, s.theta, width=bestWidth)
            trail.houghBinMax = s.binMax
            trails.append(trail)


        print "Done.", time.time() - t1

        self._mm           = mm
        self._mmCals       = mmCals
        self._isCandidate  = isCandidate
        self._brightFactor = brightFactor
        self._trails       = trails
        self._solutions    = solutions
        
        return trails

        
