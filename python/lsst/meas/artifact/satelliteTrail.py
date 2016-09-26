#!/usr/bin/env python

import numpy          as np
import copy
import matplotlib.pyplot as plt
import lsst.afw.image as afwImage

import satelliteUtils as satUtil


class SatelliteTrailList(list):
    """A container for SatelliteTrail objects.

    This inherits from a regular Python list, but includes a few useful attributes
    which are relevant to all detected trails.  There is also an additional method
    to merge other trails while avoiding duplicates.
    """
    
    def __init__(self, nPixels, binMax, psfSigma):
        """Construct a SatelliteTrailList

        @param nPixels   Total number of candidate pixels in all trails
        @param binMax    The max count among all bins in the Hough Transform
        @param psfSigma  The PSF width (as Gaussian sigma). 
        """
        
        self.nPixels = nPixels
        self.binMax = binMax
        self.psfSigma = psfSigma

        
    def cleanDuplicates(self, drMax, dThetaMax):
        """Go through this list and remove duplicates.  Return a new SatelliteTrailList."""
        
        # clean duplicates from each list
        newTrailList = SatelliteTrailList(self.nPixels, self.binMax, self.psfSigma)
        skip = []
        for i,t in enumerate(self):
            if i in skip:
                continue
            best = t
            for i2, t2 in enumerate(self[i:]):
                if t.isNear(t2, drMax, dThetaMax):
                    best = t.chooseBest(best, t2)
                    skip.append(i2)
            newTrailList.append(best)
        return newTrailList

        
    def merge(self, trailList, drMax=90.0, dThetaMax=0.15):
        """Merge trails from trailList to this SatelliteTrailList.  Returns a new SatelliteTrailList.

        @param trailList     The trailList to merge in
        @param drMax         The max separation in r for identifying duplicates
        @param dthetaMax     The max separation in theta for identifying duplicates
        """
        
        newList = SatelliteTrailList(self.nPixels, max(trailList.binMax, self.binMax), self.psfSigma)

        tList1 = trailList.cleanDuplicates(drMax, dThetaMax)
        tList2 = self.cleanDuplicates(drMax, dThetaMax)

        # get everything from list 1, and check for duplicates
        for t1 in tList1:
            best  = t1
            for t2 in tList2:
                if t1.isNear(t2, drMax, dThetaMax):
                    best = t1.chooseBest(t1, t2)
            newList.append(best)

        # get everything from list 2, and throw out duplicates (we already chose the best one)
        for t2 in tList2:
            haveIt = [t2.isNear(tNew, drMax, dThetaMax) for tNew in newList]
            if not any(haveIt):
                newList.append(t2)

        return newList

    def __str__(self):
        msg = "SatelliteTrailList(nPixels=%d, binMax=%d, psfSigma=%.2f) [%d trails]" % \
              (self.nPixels, self.binMax, self.psfSigma, len(self))
        return msg
        


##############################
# Some handy profiles to use
##############################
    
class ConstantProfile(object):
    """A constant trail profile"""

    def __init__(self, value, width):
        """Construct

        @param value  The constant value to set
        @param width  The width of the trail
        """
        self.value = value
        self.width = width
        
    def __call__(self, offset):
        """Return profile value at 'offset'

        @param offset  A distance from the center.
        """
        w  = (offset <= self.width/2.0)
        return self.value*w

    
class DoubleGaussianProfile(object):
    """A Double Gaussian trail profile"""
    
    def __init__(self, flux, sigma, fWing=0.1):
        """Construct

        @param flux       The flux of the trail
        @param sigma      Sigma of the inner Gaussian
        @param fWing      Fraction of flux in the wing (outer) Gaussian.
        """
        self.flux  = flux
        self.sigma = sigma
        self.fWing = fWing
        self.fCore = 1.0 - fWing
        self.A1  = 1.0/(np.sqrt(2.0*np.pi)*self.sigma)
        self.A2  = 1.0/(np.sqrt(2.0*np.pi)*(2.0*self.sigma))
        self.coef1 = self.flux*self.fCore*self.A1
        self.coef2 = self.flux*self.fWing*self.A2
        self.twoSigma2Core = 2.0*self.sigma**2
        self.twoSigma2Wing = 2.0*(2.0*self.sigma)**2
        
    def __call__(self, offset):
        """Return profile value at offset.

        @param offset   Distance from the profile center.
        """
        g1  = np.exp(-offset**2/self.twoSigma2Core)
        g2  = np.exp(-offset**2/self.twoSigma2Wing)
        out = self.coef1*g1 + self.coef2*g2
        return out

        
class SatelliteTrail(object):
    """Hold parameters related to a satellite trail.

    Parameters are stored in Hesse normal form (r, theta).  The trail also
    knows its width, and its flux, and provides method to insert itself into an image
    or to set mask bits in an exposure.
    """
    
    def __init__(self, r, theta, width=0.0, flux=1.0, center=0.0, binMax=None, resid=None, maskAndBits=None):
        """Construct a SatelliteTrail with specified parameters.

        @param r        r from Hesse normal form of the trail
        @param theta    theta from Hesse normal form of the trail
        @param width    The width of the trail (0 for a PSF, out-of-focus aircraft are wider)
        @param flux     Flux of the trail
        @param binMax   The max bin count for the Hough solution
        @param resid    The coordinate resid tuple (median,inter_quart_range) for residuals from the solution
        @param maskAndBits  Only mask pixels with these bits set.
        """

        self._r       = r
        self._theta   = theta
        self.vx       = np.cos(theta)
        self.vy       = np.sin(theta)
        
        self.flux     = flux
        self.width    = width
        self.center   = center

        self.binMax   = binMax
        self.resid    = resid

        self.detectWidth = None
        self.maskAndBits  = maskAndBits
        
        self.nMaskedPixels  = 0

    @property
    def r(self):
        return self._r
    @r.setter
    def r(self, value):
        self._r = value
    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, value):
        self._theta = value
        self.vx = np.cos(value)
        self.vy = np.sin(value)

        
    @classmethod
    def fromHoughSolution(cls, solution, bins):
        """A constructor to create a SatelliteTrail from a HoughSolution Object

        @param solution      The HoughSolution from which to construct ourself.
        @param bins          Binning used in solution image.
        """
        
        trail = cls(bins*solution.r, solution.theta, binMax=solution.binMax, resid=solution.resid)
        return trail
        
        
    def setMask(self, exposure, satelliteBit, andMask=None):
        """Set the mask plane near this trail in an exposure.

        @param exposure      The exposure with mask plane to be set.  Change is in-situ.
        @param satelliteBit  The bit to use for this mask.
        
        @return nPixels    The number of pixels set.
        """

        msk            = exposure.getMaskedImage().getMask()
        
        # create a fresh mask and add to that.
        tmp            = type(msk)(msk.getWidth(), msk.getHeight())
        sigma          = satUtil.getExposurePsfSigma(exposure)
        # if this is being called, we probably have a width measured with our measure() method

        # never narrower than the PSF
        if self.width < 2.0*sigma:
            # go to +- 10*sigma to get the extended wings
            width = 20.0*sigma
        else:
            # or 15x what was measured
            width = 15.0*self.width
        profile        = ConstantProfile(satelliteBit, width)
        self.insert(tmp, profile, width)

        # only set bits which already have 'andMask' set.
        # This is intended to allow only 'DETECTED' pixels to be set.  I don't know what other bit
        # might be useful, but a mask was chosen as it's no harder to implement.
        if self.maskAndBits:
            andBits = 0
            for label in self.maskAndBits:
                andBits |= msk.getPlaneBitMask(label)
            whereDetectMask   = (msk.getArray() & andBits) > 0
            whereSatAndDetect = (tmp.getArray() > 0) & whereDetectMask
            tmp.getArray()[~whereSatAndDetect] = 0
        
        # OR it in to the existing plane, return the number of pixels we set
        msk     |= tmp
        nPixels  = (tmp.getArray() > 0).sum()
        self.nMaskedPixels += nPixels
        return nPixels

    def endPoints(self, nx, ny): 
        """Compute the length of the trail in an nx*ny image.

        @param nx      Image x dimension
        @param ny      Image y dimension
        """

        points = []
        epsilon = 1.0e-8
        if np.abs(self.vy) > epsilon:
            for ix in 0, nx:
                y = (self.r - ix*self.vx)/self.vy
                if (y >= 0) and (y < ny):
                    points.append( (ix, y) )
        if np.abs(self.vx) > epsilon:
            for iy in 0, ny:
                x = (self.r - iy*self.vy)/self.vx
                if (x >= 0) and (x < nx):
                    points.append( (x, iy) )
        points = sorted(points, key=lambda x: x[0])
        if len(points) != 2:
            points = ( (0,0), (0,0) )
        return points
    
    def length(self, nx, ny):
        """Compute the length of the trail in an nx*ny image.

        @param nx      Image x dimension
        @param ny      Image y dimension
        """

        p1, p2 = self.endPoints(nx, ny)
        length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
        return length
        
    
    def trace(self, nx, ny, offset=0, bins=1):
        """Get x,y values near this satellite trail.

        @param nx      Image width in pixels
        @param ny      Image height in pixels
        @param offset  Distance from trail centerline to return values
        @param bins    Correct for images binned by this amount.
        """

        p1, p2 = self.endPoints(nx, ny)
        xs = p1[0], p2[0]
        ys = p1[1], p2[1]

        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        if dx > dy:
            lo, hi = min(xs), max(xs)
            x = np.arange(lo, hi)
            y = (self.r/bins + offset - x*self.vx)/self.vy
        else:
            lo, hi = min(ys), max(ys)
            y = np.arange(lo, hi)
            x = (self.r/bins + offset - y*self.vy)/self.vx

        return np.rint(x).astype(int), np.rint(y).astype(int)

    
    def shiftOrigin(self, dx, dy):
        dr       = dx*self.vx + dy*self.vy
        rNew     = self.r - dr
        thetaNew = self.theta
        if rNew < 0.0:
            rNew *= -1.0
            thetaNew += np.pi if thetaNew < np.pi else -np.pi
        trail = copy.copy(self)
        trail.r = rNew
        trail.theta = thetaNew
        return trail
        
    def residual(self, x, y, bins=1):
        """Get residuals of this fit compared to given x,y coords.

        @param x   array of x pixel coord
        @param y   array of y pixel coord
        """

        dr = x*self.vx + y*self.vy - self.r/bins
        return dr

        
    def insert(self, exposure, profile, width):
        """Plant this satellite trail in a given exposure.

        @param exposure       The exposure to plant in (accepts ExposureF, ImageF, MaskU or ndarray)
        @param profile        A profile function object.
        @param width          Set pixels to this value.  (Don't plant a Double-Gaussian trail profile).

        This method serves a few purposes.

        (1) To search for a trail with profile similar to a PSF, we plant a PSF-shaped trail
            and measure its parameters for use in CALIBRATING detection limits.
        (2) When we find a trail, our setMask() method calls this method with a maskBit to SET THE MASK.
        (3) For TESTING, we can insert fake trails and try to find them.
        """

        # Handle Exposure, Image, ndarray
        if isinstance(exposure, afwImage.ExposureF):
            img = exposure.getMaskedImage().getImage().getArray()
        elif isinstance(exposure, afwImage.ImageF):
            img = exposure.getArray()
        elif isinstance(exposure, afwImage.MaskU):
            img = exposure.getArray()
        elif isinstance(exposure, np.ndarray):
            img = exposure
            
        ny, nx = img.shape

        #############################
        # plant the trail
        #############################
        xpart, ypart = np.arange(nx)*self.vx, np.arange(ny)*self.vy
        dot = np.add.outer(ypart, xpart)

        # plant the trail using the distance from our line
        # as the parameter in a 1D DoubleGaussian
        offset = np.abs(dot - self.r)

        # only bother updating the pixels within width
        w = (offset < width/2.0)
        #img += profile(offset)
        img[w] += profile(offset[w]).astype(img.dtype)
        return img


    def measure(self, exposure, bins=1, widthIn=None):
        """Measure an aperture flux, a centroid, and a width for this satellite trail in a given exposure.

        @param exposure       The exposure to measure in (accepts ExposureF, ImageF, ndarray)
        @param bins           The binning used in the given exposure.  Needed as r is in pixels.
        @param widthIn        The aperture within which to measure.  Default = existing width

        For a PSF trail, our width is 0.0.  A call with an exposure will use the PSF and
        select 4*psfSigma as a widthIn value.
        """

        aperture = self.width
        
        # Handle Exposure, Image, ndarray
        if isinstance(exposure, afwImage.ExposureF):
            img = exposure.getMaskedImage().getImage().getArray()
            nx, ny = exposure.getWidth(), exposure.getHeight()
            
            # If we're a PSF trail, our width is 0.0 ... use an aperture based on the PSF
            if np.abs(self.width) < 1.0e-6:
                aperture = 4.0*satUtil.getExposurePsfSigma(exposure, minor=True)

        elif isinstance(exposure, afwImage.ImageF):
            img = exposure.getArray()
            nx, ny = exposure.getWidth(), exposure.getHeight()

        elif isinstance(exposure, np.ndarray):
            img = exposure
            ny, nx = img.shape

        # always obey the user
        if widthIn:
            aperture = widthIn

        # Only ExposureF knows its PSF width, so we might not have an aperture
        if aperture is None:
            raise ValueError("Must specify width for satellite trail flux")

            
        #############################
        # plant the trail
        #############################
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

        # plant the trail using the distance from our line
        # as the parameter in a 1D DoubleGaussian
        dot    = xx*self.vx + yy*self.vy
        offset = dot - self.r/bins

        hwidth = aperture/2.0

        # only bother updating the pixels within 5-sigma of the line
        w = (np.abs(offset) < hwidth) & (np.isfinite(img))
        self.flux     = img[w].sum()
        self.center   = bins*(img[w]*offset[w]).sum()/self.flux
        sigma         = np.sqrt((img[w]*offset[w]**2).sum()/self.flux)

        # iterate once to suppress noise
        off2 = offset[w]**2
        weight = np.exp(-0.5*off2/(2.0*sigma)**2)/(2.0*np.pi*np.sqrt(2.0*sigma))
        sigma         = bins*np.sqrt((weight*img[w]*off2).sum()/self.flux)
            
        self.width    = 2.0*sigma

        return self.flux

        
    def __str__(self):
        med, iqr = -1.0, -1.0
        if self.resid is not None:
            med,iqr = self.resid
        rep = "SatelliteTrail(r=%.1f,theta=%.3f,width=%.2f,flux=%.2f,binMax=%r,resid=(%.2f,%.2f))" % \
              (self.r, self.theta, self.width, self.flux, self.binMax, med, iqr)
        return rep
        
    def __repr__(self):
        rep = "SatelliteTrail(r=%r,theta=%r,width=%r,flux=%r,center=%r,binMax=%r,resid=(%r))" % \
              (self.r, self.theta, self.width, self.flux, self.center, self.binMax, self.resid)
        return rep

    def __eq__(self, trail):
        isEq = (self.r == trail.r) and (self.theta == trail.theta) and \
               (self.width == trail.width) and (self.flux == trail.flux)
        return isEq
        
    def isNear(self, trail, drMax, dThetaMax):
        """Fuzzy-compare two trails.

        It's quite possible that the same trail will be detected in a searches for satellites and
        aircraft.  The parameters won't be identical, but they'll be close.

        Use dot product to handle the wrap at 2*pi
        """
        r1, r2 = self.r,     trail.r
        t1, t2 = self.theta, trail.theta
        rTest = np.abs(r2 - r1) < drMax
        tTest = satUtil.angleCompare(t1, t2, dThetaMax)
        return rTest and tTest

        
    @staticmethod
    def chooseBest(trail1, trail2):
        """A single place to choose the best trail, if two solutions exist.

        @param trail1   SatelliteTrail object #1
        @param trail2   SatelliteTrail object #2
        """
        err1 = trail1.resid.iqr
        err2 = trail2.resid.iqr
        return trail1 if (err1 < err2) else trail2
        
