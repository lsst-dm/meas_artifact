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


def displayArray(array, frame, title=None):
    import lsst.afw.display.ds9 as ds9
    typemap = {np.dtype(t1): t2 for t1, t2 in
               (("int32", afwImage.ImageI),
                ("bool", afwImage.ImageU),
                ("float32", afwImage.ImageF))}

    dummy = typemap[array.dtype](*reversed(array.shape))
    dummy.getArray()[:] = array
    ds9.mtv(dummy, frame=frame, title=title)


class SatelliteFinder(object):
    """Class to find satellite trails and other linear features in images.

    This uses a modified Hough Transform.  Briefly, the images are binned, and convolved with
    kernels to solve for the local alignment of flux near each pixel (like getting an adaptive
    moment at each pixel).  From this, a local position angle can be computed and used
    in a Hough Transform (rather than testing all possible thetas).

    In addition to a reasonable estimate of theta, the ratio of moments along/perpendicular to
    the local alignment (i.e. like an ellipticity) has a known value based on the PSF width.
    By using a 'calibration trail', a fake satellite trail can be created and measured to
    provide expected values for trail width 'b', 'ellipticity' e = 1 - b/a, and other
    moment-based quantities.  These expected values are used to cull the herd of pixels down
    to a more manageable number.

    If the number of candidate pixels is small enough, they can be compared pairwise to further
    weed out non-satellite-trail points.  With two points, we have two local estimates of theta
    from the convolution of moments, but also a direct delta-x,delta-y estimate.  If both points
    in the pair are part of the same trail theta1,theta2 should be similar and should match the
    thetaXY based on the pixel coordinates.  Coincident theta1==theta2==thetaXY points are kept.
    This is done by the thetaAlignment() routine.
    
    Finally, in computing the Hough transform, the input thetas are still quite noisy, but
    the functional form in Hough space (r,theta space) is known: r = x*cos(theta) + y*sin(theta).
    The derivative is easily computable and each r,theta point can be extrapolated along its
    derivative as a tangent line.  The intersection point of these tangent extrapolations gives a
    very robust estimate of theta.

    """
    
    def __init__(self,
                 bins            = 4,
                 doBackground    = True,
                 scaleDetected   = 10.0,
                 sigmaSmooth     = 1.0,
                 thetaTolerance  = 0.15,
                 
                 luminosityLimit = 0.02,       
                 centerLimit     = 1.2,       
                 eRange          = 0.08,       
                 bLimit          = 1.4,
                 skewLimit       = 10.0,       
                 
                 kernelSigma     = 7,     
                 kernelWidth     = 11,    
                 growKernel      = 1.4,
                 
                 houghBins       = 200,        
                 houghThresh     = 40,
                 
                 maxTrailWidth   = 2.1,
                 maskAndBits     = (),
                 
                 log             = None,
                 verbose         = False,
             ):
        """Construct SatelliteFinder

        @param bins              Binning to use (improves speed, but becomes unreliable above 4)
        @param doBackground      Subtract median-ring filter background
        @param scaleDetected     Scale pixels with detected flag by this amount.
        @param growKernel        Repeat with a kernel larger by this fraction (no repeat if 1.0)
        @param sigmaSmooth       Do a Gaussian smooth with this sigma (binned pixels)
        @param thetaTolerance    Max theta difference for thetaAlignment() routine.
        
        @param luminosityLimit   Min flux to accept  [units of Std.Dev].
        @param centerLimit       Max error in 1st moment (centroid) to accept [pixels].
        @param eRange            Max error in e=1-b/a above and below the calib trail value.
        @param bLimit            Max error in trail width [pixels].
        @param skewLimit         Max error in 3rd moment (skewness) to accept [pixels^3]
        
        @param kernelSigma       Gaussian sigma to taper the kernel [pixels]
        @param kernelWidth       Width of kernel in pixels.
        
        @param houghBins         Number of bins in r,theta space (total = bins x bins)
        @param houghThresh       Count level in Hough bin to consider a detection.
        
        @param maxTrailWidth     Discard trail detections wider than this.
        @param maskAndBits       Only allow pixels with these bits to be masked (typ. DETECTED, if any)
        
        @param log               A log object.
        @param verbose           Be chatty.
        """
        
        self.bins              = bins
        self.doBackground      = doBackground
        self.scaleDetected     = scaleDetected
        self.sigmaSmooth       = sigmaSmooth
        self.thetaTolerance    = thetaTolerance
        
        self.kernelSigma       = kernelSigma        
        self.kernelWidth       = kernelWidth
        self.growKernel        = 1.4

        self.centerLimit       = centerLimit
        self.eRange            = eRange
        self.houghThresh       = houghThresh
        self.houghBins         = houghBins
        self.luminosityLimit   = luminosityLimit
        self.skewLimit         = skewLimit
        self.bLimit            = bLimit

        self.maxTrailWidth     = maxTrailWidth
        self.maskAndBits       = maskAndBits
        
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
            insertWidth = self.bins*kernelWidth # 4.0*(width/2.0 + psfSigma)
        calTrail.insert(calArr, profile, insertWidth)

        if False:
            displayArray(calArr, frame=2, title="Calibration image, unsmoothed")
            import pdb;pdb.set_trace()

        # Now bin and smooth, just as we did the real image
        calArr   = afwMath.binImage(calImg, self.bins).getArray()
        calArr   = satUtil.smooth(calArr, self.sigmaSmooth)

        if False:
            displayArray(calArr, frame=2, title="Calibration image, smoothed")
            import pdb;pdb.set_trace()

        return calArr

        
    def getTrails(self, exposure, widths):
        """Detect satellite trails in exposure using provided widths

        @param exposure      The exposure to detect in
        @param widths        A list of widths [pixels] to use for calibration trails.

        @return trails       A SatelliteTrailList object containing detected trails.
        """

        emsk = exposure.getMaskedImage().getMask()
        DET  = emsk.getPlaneBitMask("DETECTED")
        MASK = 0
        for plane in "BAD", "CR", "SAT", "INTRP", "EDGE", "SUSPECT":
            MASK |= emsk.getPlaneBitMask(plane)
        
        t1 = time.time()

        import pdb;pdb.set_trace()

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

        # scale the detected pixels
        if np.abs(self.scaleDetected - 1.0) > 1.0e-6:
            wDet       = msk & DET > 0
            sig = imgClip.std()
            wSig = img > 2.0*sig
            # amplify detected pixels (make this configurable?)
            img[wDet|wSig] *= self.scaleDetected

        # subtract a small scale background when we search for PSFs
        if self.doBackground:
            back       = satUtil.medianRing(imgClip, self.kernelWidth, 2.0*self.sigmaSmooth)
            img       -= back
            imgClip   -= back

        
        #   - smooth 
        img       = satUtil.smooth(img,       self.sigmaSmooth)
        imgClip   = satUtil.smooth(imgClip, self.sigmaSmooth)
        rms       = imgClip[(mskClip & (MASK | DET) == 0)].std()

        if True:
            import lsst.afw.display.ds9 as ds9
            ds9.mtv(exp, frame=1, title="Image")
        
        isCandidate = np.ones(img.shape, dtype=bool)

        ###########################################
        # Try different kernel sizes
        ###########################################
        
        mmCals = []
        nHits = []

        # Different sized kernels should give the same results for a real trail
        # but would be less likely to for noise.
        # Unfortunately, this is costly, and the effect is small.
        for kernelFactor in (1.0, self.growKernel):
            self.log.info("Getting moments with kernel grown by factor of %.1f" % (kernelFactor,))
            kernelWidth = 2*int((kernelFactor*self.kernelWidth)//2) + 1
            kernelSigma = kernelFactor*self.kernelSigma 

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

            if True:
                displayArray(mm.imageMoment.ix, frame=2, title="Ix")
                displayArray(mm.imageMoment.iy, frame=3, title="Iy")
                ellip, theta, b = momCalc.momentToEllipse(mm.imageMoment.ixx, mm.imageMoment.iyy, mm.imageMoment.ixy)
                displayArray(ellip, frame=4, title="Ellip")
                displayArray(b, frame=5, title="b")
                displayArray(np.sqrt(mm.imageMoment.ixxx**2 + mm.imageMoment.iyyy**2), frame=6, title="Skew")
                print "KernelFactor=%f bins=%d" % (kernelFactor, self.bins)
                import pdb;pdb.set_trace()


            #Selector = momCalc.PixelSelector
            Selector = momCalc.PValuePixelSelector
            maxPixels = 4000
            for i, calImg in enumerate(calImages):
                mmCal = momCalc.MomentManager(calImg, kernelWidth=kernelWidth, kernelSigma=kernelSigma, 
                                              isCalibration=True)

                self.log.info("Width=%f kernelFactor=%f bins=%d: sumI=%f center=%f centerPerp=%f skew=%f skewPerp=%f ellip=%f b=%f" %
                              (widths[i], kernelFactor, self.bins, mmCal.sumI, mmCal.center, mmCal.center_perp, mmCal.skew, mmCal.skew_perp, mmCal.ellip, mmCal.b))

                mmCals.append(mmCal)

                sumI  = momCalc.MomentLimit('sumI',        self.luminosityLimit*rms, 'lower')
                cent  = momCalc.MomentLimit('center',      2.0*self.centerLimit,     'center')
                centP = momCalc.MomentLimit('center_perp', self.centerLimit,         'center')
                skew  = momCalc.MomentLimit('skew',        2.0*self.skewLimit,       'center')
                skewP = momCalc.MomentLimit('skew_perp',   self.skewLimit,           'center')
                ellip = momCalc.MomentLimit('ellip',       self.eRange,              'center')
                b     = momCalc.MomentLimit('b',           self.bLimit,              'center')

                selector = Selector(mm, mmCal)
                for limit in (
                    sumI,
                    ellip,
                    cent,
                    centP,
                    skew,
                    skewP,
                    b
                    ):
                    selector.append(limit)

                pixels      = selector.getPixels(maxPixels=maxPixels)

                if True:
                    displayArray(pixels, frame=7, title="Selected pixels for width")
                    print "Width=%f kernelFactor=%f bins=%d" % (widths[i], kernelFactor, self.bins)
                    import pdb;pdb.set_trace()


                isKernelCandidate |= pixels

                nHits.append((widths[i], isKernelCandidate.sum()))

                msg = "cand: nPix: %d  tot: %d" % (pixels.sum(), isKernelCandidate.sum())
                self.log.info(msg)


            isCandidate &= isKernelCandidate
            self.log.info("total: %d" % (isCandidate.sum()))

            if True:
                displayArray(isCandidate, frame=8, title="Selected pixels so far")
                import pdb;pdb.set_trace()

       
        bestCal = sorted(nHits, key=lambda x: x[1], reverse=True)[0]
        bestWidth = bestCal[0]

        ###############################################
        # Theta Alignment
        ###############################################
        xx, yy = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))
        nBeforeAlignment = isCandidate.sum()
        maxSeparation = min([x/2 for x in img.shape])
        thetaMatch, newTheta = hough.thetaAlignment(mm.theta[isCandidate], xx[isCandidate], yy[isCandidate],
                                                    tolerance=self.thetaTolerance,
                                                    limit=3, maxSeparation=maxSeparation)

        mm.theta[isCandidate] = newTheta
        isCandidate[isCandidate] = thetaMatch
        nAfterAlignment = isCandidate.sum()
        self.log.info("theta-alignment Bef/aft: %d / %d" % (nBeforeAlignment, nAfterAlignment))

        if True:
            displayArray(isCandidate, frame=9, title="Selected pixels for Hough")

        #################################################
        # Hough transform
        #################################################
        rMax           = np.linalg.norm(img.shape)
        houghTransform = hough.HoughTransform(self.houghBins, self.houghThresh, rMax=rMax,
                                              maxPoints=10000, nIter=1, maxResid=5.5, log=self.log)
        solutions      = houghTransform(mm.theta[isCandidate], xx[isCandidate], yy[isCandidate])

        #################################################
        # Construct Trail objects from Hough solutions
        #################################################
        trails = satTrail.SatelliteTrailList(nAfterAlignment, solutions.binMax, psfSigma)
        for s in solutions:
            trail = satTrail.SatelliteTrail.fromHoughSolution(s, self.bins)
            trail.detectWidth = bestWidth
            trail.maskAndBits = self.maskAndBits
            trail.measure(exp, bins=self.bins)
            # last chance to drop it
            if trail.width < self.maxTrailWidth:
                nx, ny = exposure.getWidth(), exposure.getHeight()
                self.log.info(str(trail) + "[%dpix]" % (trail.length(nx,ny)))
                trails.append(trail)
            else:
                self.log.info("Dropping (maxWidth>%.1f): %s" %(self.maxTrailWidth, trail))


        # A bit hackish, but stash some useful info for diagnostic plots
        self._mm           = mm
        self._mmCals       = mmCals
        self._isCandidate  = isCandidate
        self._brightFactor = 10
        self._trails       = trails
        self._solutions    = solutions
        
        return trails

        
