#!/usr/bin/env python

import numpy          as np

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

    def merge(self, trailList, drMax=40.0, dThetaMax=0.15):
        """Merge trails from trailList to this SatelliteTrailList.  Returns a new SatelliteTrailList.

        @param trailList     The trailList to merge in
        @param drMax         The max separation in r for identifying duplicates
        @param dthetaMax     The max separation in theta for identifying duplicates
        """
        
        s = SatelliteTrailList(self.nPixels, max(trailList.binMax, self.binMax), self.psfSigma)

        # get everything from list 1, and check for duplicates
        for t in trailList:
            best  = t
            for t2 in self:
                if t.isNear(t2, drMax, dThetaMax):
                    best = t.chooseBest(t, t2)
            s.append(best)

        # get everything from list 2, and throw out duplicates (we already chose the best one)
        for t in self:
            haveIt = [t.isNear(t2, drMax, dThetaMax) for t2 in s]
            if not any(haveIt):
                s.append(t)

        return s

    def __str__(self):
        msg = "SatelliteTrailList(nPixels=%d, binMax=%d, psfSigma=%.2f)" % (self.nPixels, self.binMax, self.psfSigma)
        return msg
        
        
class ConstantProfile(object):
    def __init__(self, value, width):
        self.value = value
        self.width = width
    def __call__(self, offset):
        w  = (offset < self.width/2)
        out = np.zeros(offset.shape, dtype=type(self.value))
        out[w] = self.value
        return out
        
class DoubleGaussianProfile(object):
    def __init__(self, flux, sigma, fWing=0.1):
        self.flux  = flux
        self.sigma = sigma
        self.fWing = fWing
        self.fCore = 1.0 - fWing
        
    def __call__(self, offset):
        A1  = 1.0/(2.0*np.pi*self.sigma**2)
        g1  = np.exp(-offset**2/(2.0*self.sigma**2))
        A2  = 1.0/(2.0*np.pi*(2.0*self.sigma)**2)
        g2  = np.exp(-offset**2/(2.0*(2.0*self.sigma)**2))
        out = self.flux*(self.fCore*A1*g1 + self.fWing*A2*g2)
        return out

        
class SatelliteTrail(object):
    """Hold parameters related to a satellite trail.

    Parameters are stored in Hesse normal form (r, theta).  The trail also
    knows its width, and its flux, and provides method to insert itself into an image
    or to set mask bits in an exposure.
    """
    
    def __init__(self, r, theta, width=0.0, flux=1.0, center=0.0, binMax=None, resid=None):
        """Construct a SatelliteTrail with specified parameters.

        @param r        r from Hesse normal form of the trail
        @param theta    theta from Hesse normal form of the trail
        @param width    The width of the trail (0 for a PSF, out-of-focus aircraft are wider)
        @param flux     Flux of the trail
        @param binMax   The max bin count for the Hough solution
        @param resid    The coordinate resid tuple (median,inter_quart_range) for residuals from the solution
        """

        self.r        = r
        self.theta    = theta
        self.vx       = np.cos(theta)
        self.vy       = np.sin(theta)
        
        self.flux     = flux
        self.width    = width
        self.center   = center

        self.binMax   = binMax
        self.resid    = resid
        
    @classmethod
    def fromHoughSolution(cls, solution, bins):
        """A constructor to create a SatelliteTrail from a HoughSolution Object

        @param solution      The HoughSolution from which to construct ourself.
        @param bins          Binning used in solution image.
        """
        
        trail = cls(bins*solution.r, solution.theta, binMax=solution.binMax, resid=solution.resid)
        return trail
        
        
    def setMask(self, exposure):
        """Set the mask plane near this trail in an exposure.

        @param exposure    The exposure with mask plane to be set.  Change is in-situ.

        @return nPixels    The number of pixels set.
        """
        
        # add a new mask plane
        msk            = exposure.getMaskedImage().getMask()
        satellitePlane = msk.addMaskPlane("SATELLITE")
        satelliteBit   = 1 << satellitePlane

        # create a fresh mask and add to that.
        tmp            = type(msk)(msk.getWidth(), msk.getHeight())
        sigma          = satUtil.getExposurePsfSigma(exposure)
        # if this is being called, we probably have a width measured with our measure() method

        width          = 8.0*self.width
        profile        = ConstantProfile(satelliteBit, width)
        self.insert(tmp, profile, width)

        # OR it in to the existing plane, return the number of pixels we set
        msk     |= tmp
        nPixels  = (tmp.getArray() > 0).sum()
        return nPixels

        
    def trace(self, nx, ny, offset=0, bins=1):
        """Get x,y values near this satellite trail.

        @param nx      Image width in pixels
        @param ny      Image height in pixels
        @param offset  Distance from trail centerline to return values
        @param bins    Correct for images binned by this amount.
        """
        
        x = np.arange(nx)
        y = (self.r/bins + offset - x*self.vx)/self.vy
        w =  (x > 0) & (x < nx) & (y > 0) & (y < ny)
        return x[w], y[w]


    def residual(self, x, y, bins=1):
        """Get residuals of this fit compared to given x,y coords.

        @param x   array of x pixel coord
        @param y   array of y pixel coord
        """

        dr = x*cos(t) + y*sin(t) - self.r/bins
        return dr

        
    def insert(self, exposure, profile, width):
        """Plant this satellite trail in a given exposure.

        @param exposure       The exposure to plant in (accepts ExposureF, ImageF, MaskU or ndarray)
        @param sigma          The PSF size (as Gaussian sigma) to use.
        @param maskBit        Set pixels to this value.  (Don't plant a Double-Gaussian trail profile).

        This method serves a few purposes.

        (1) To search for a trail with profile similar to a PSF, we plant a PSF-shaped trail
            and measure its parameters for use in calibrating detection limits.
        (2) When we find a trail, our setMask() method calls this method with a maskBit to set.
        (3) For testing, we can insert fake trails and try to find them.
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
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

        # plant the trail using the distance from our line
        # as the parameter in a 1D DoubleGaussian
        dot    = xx*self.vx + yy*self.vy
        offset = np.abs(dot - self.r)

        # only bother updating the pixels within 5-sigma of the line
        w = (offset < width/2)
        img[w] = profile(offset[w])
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
        offset = np.abs(dot - self.r/bins)

        hwidth = aperture/2.0

        # only bother updating the pixels within 5-sigma of the line
        w = (offset < hwidth) & (np.isfinite(img))
        self.flux     = img[w].sum()
        self.center   = bins*(img[w]*offset[w]).sum()/self.flux
        sigma         = bins*np.sqrt((img[w]*offset[w]**2).sum()/self.flux)
        self.width    = 2.0*sigma

        return self.flux

        
    def __str__(self):
        rep = "SatelliteTrail(r=%.1f,theta=%.3f,width=%.2f,flux=%.2f,binMax=%d,resid=(%.2f,%.2f))" % \
              (self.r, self.theta, self.width, self.flux, self.binMax, self.resid.med, self.resid.iqr)
        return rep
        
    def __repr__(self):
        rep = "SatelliteTrail(r=%r,theta=%r,width=%r,flux=%r,center=%r,fWing=%r,binMax=%r,resid=(%r))" % \
              (self.r, self.theta, self.width, self.flux, self.center, self.fWing, self.binMax, self.resid)
        return rep

    def __eq__(self, trail):
        isEq = (self.r == trail.r) and (self.theta == trail.theta) and \
               (self.width == trail.width) and (self.flux == trail.flux) and (self.fWing == trail.fWing)
        return isEq
        
    def isNear(self, trail, drMax, dThetaMax):
        """Fuzzy-compare two trails.

        It's quite possible that the same trail will be detected in a searches for satellites and
        aircraft.  The parameters won't be identical, but they'll be close.
        """
        isNear = (np.abs(self.r - trail.r) < drMax) and (np.abs(self.theta - trail.theta) < dThetaMax)
        return isNear

    @staticmethod
    def chooseBest(trail1, trail2):
        """A single place to choose the best trail, if two solutions exist.

        @param trail1   SatelliteTrail object #1
        @param trail2   SatelliteTrail object #2
        """
        err1 = trail1.resid.iqr
        err2 = trail2.resid.iqr
        return trail1 if (err1 < err2) else trail2
        
