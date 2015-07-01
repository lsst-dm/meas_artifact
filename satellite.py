#!/usr/bin/env python

import os
import time, datetime
import collections

import numpy as np
import numpy.fft as fft

import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
from matplotlib.patches import Rectangle

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as ellipses
import lsst.afw.math as afwMath

import hough
import satellite_utils as satUtil
import hesse_cluster as hesse
import momentCalculator as momCalc
import satelliteDebug as satDebug

class SatelliteTrailList(list):
    def __init__(self, nPixels, binMax, psfSigma):
        self.nPixels = nPixels
        self.binMax = binMax
        self.psfSigma = psfSigma

        self.drMax = 50.0
        self.dthetaMax = 0.1
        
    def merge(self, trailList):

        s = SatelliteTrailList(self.nPixels, max(trailList.binMax, self.binMax), self.psfSigma)
        for t in self:
            s.append(t)
            
        for t in trailList:
            r     = t.r
            theta = t.theta

            isDuplicate = False
            for t2 in self:
                dr = t2.r - r
                dt = t2.theta - theta
                if abs(dr) < self.drMax and abs(dt) < self.dthetaMax:
                    isDuplicate = True
            if not isDuplicate:
                s.append(t)
        return s

class SatelliteTrail(object):
    def __init__(self, r, theta, width=None, flux=1.0, f_wing=0.1):
        if width is None:
            width = [0.0]
        self.r     = r
        self.theta = theta
        self.width = width
        self.vx    = np.cos(theta)
        self.vy    = np.sin(theta)
        self.flux  = flux
        self.f_core = 1.0 - f_wing
        self.f_wing = f_wing

        self.houghBinMax = 0
        
    def setMask(self, exposure):

        msk = exposure.getMaskedImage().getMask()
        sigma = satUtil.getExposurePsfSigma(exposure)
        satellitePlane = msk.addMaskPlane("SATELLITE")
        satelliteBit = 1 << satellitePlane
        tmp = type(msk)(msk.getWidth(), msk.getHeight())
        self.insert(tmp, sigma=sigma, maskBit=satelliteBit)
        msk |= tmp
        # return the number of masked pixels
        return len(np.where(tmp.getArray() > 0)[0])
        
    def trace(self, nx, ny, offset=0, bins=1):
        x = np.arange(nx)
        y = (self.r/bins + offset - x*self.vx)/self.vy
        w, = np.where( (x > 0) & (x < nx) & (y > 0) & (y < ny) )
        return x[w], y[w]


    def insert(self, exposure, sigma=None, maskBit=None):

        if sigma and sigma < 1.0:
            sigma = 1.0
        
        # Handle Exposure, Image, ndarray
        if isinstance(exposure, afwImage.ExposureF):
            img = exposure.getMaskedImage().getImage().getArray()
            nx, ny = exposure.getWidth(), exposure.getHeight()
            if sigma is None:
                sigma = satUtil.getExposurePsfSigma(exposure, minor=True)
                
        elif isinstance(exposure, afwImage.ImageF):
            img = exposure.getArray()
            nx, ny = exposure.getWidth(), exposure.getHeight()

        elif isinstance(exposure, afwImage.MaskU):
            img = exposure.getArray()
            nx, ny = exposure.getWidth(), exposure.getHeight()
        elif isinstance(exposure, np.ndarray):
            img = exposure
            ny, nx = img.shape

        if sigma is None:
            raise ValueError("Must specify sigma for satellite trail width")

            
        #############################
        # plant the trail
        #############################
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

        # plant the trail using the distance from our line
        # as the parameter in a 1D DoubleGaussian
        dot    = xx*self.vx + yy*self.vy
        offset = np.abs(dot - self.r)

        # go to 4 sigma of the wider gaussian in the double gauss
        if self.width > 1:
            hwidth = self.width/2.0
        else:
            hwidth = 8.0*sigma

        # only bother updating the pixels within 5-sigma of the line
        wy,wx  = np.where(offset < hwidth)
        if maskBit:
            img[wy,wx] = maskBit
        else:
            A1  = 1.0/(2.0*np.pi*sigma**2)
            g1  = np.exp(-offset[wy,wx]**2/(2.0*sigma**2))
            A2  = 1.0/(2.0*np.pi*(2.0*sigma)**2)
            g2  = np.exp(-offset[wy,wx]**2/(2.0*(2.0*sigma)**2))
            img[wy,wx] += self.flux*(self.f_core*A1*g1 + self.f_wing*A2*g2)
        
        return img

        

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
        calTrail = SatelliteTrail(cx, cy, width=width)
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
            fig.savefig("junk.png")

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
        UNWANT = MASK | DET
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
                'sumI'         : -2.5*rms_faint,
                #'center'       : 8.0*self.centerLimit,
                #'center_perp'  : 4.0*self.centerLimit,
                #'skew'         : 8.0*self.skewLimit,
                #'skew_per'     : 4.0*self.skewLimit,
                #'ellip'        : 8.0*self.eRange,
                #'b'            : 2.0,
            }                
            selector = Selector(mm_faint, mmCal, faint_limits)
            faint_pixels = selector.getPixels() & (msk & (MASK | DET) == 0)
            #pixels |= faint_pixels

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
        trails = SatelliteTrailList(nCandidatePixels, solutions.binMax, psfSigma)
        for s in solutions:
            print "Trail: ", self.bins*s.r, s.theta, bestWidth, s.binMax
            trail = SatelliteTrail(self.bins*s.r, s.theta, width=bestWidth)
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

        
