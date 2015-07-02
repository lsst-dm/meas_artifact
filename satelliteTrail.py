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

    def merge(self, trailList, drMax=50.0, dthetaMax=0.1):
        """Merge trails from trailList to this SatelliteTrailList.  Returns a new SatelliteTrailList.

        @param trailList     The trailList to merge in
        @param drMax         The max separation in r for identifying duplicates
        @param dthetaMax     The max separation in theta for identifying duplicates
        """
        
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
                if abs(dr) < drMax and abs(dt) < dthetaMax:
                    isDuplicate = True
            if not isDuplicate:
                s.append(t)
        return s


class SatelliteTrail(object):
    """Hold parameters related to a satellite trail.

    Parameters are stored in Hesse normal form (r, theta).  The trail also
    knows its width, and its flux, and provides method to insert itself into an image
    or to set mask bits in an exposure.
    """
    
    def __init__(self, r, theta, width=0.0, flux=1.0, fWing=0.1):
        """Construct a SatelliteTrail with specified parameters.

        @param r        r from Hesse normal form of the trail
        @param theta    theta from Hesse normal form of the trail
        @param width    The width of the trail (0 for a PSF, out-of-focus aircraft are wider)
        @param flux     Flux of the trail
        @param fWing    For double-Gaussian PSF model.  Fraction of flux in larger Gaussian.
        """

        self.r     = r
        self.theta = theta
        self.width = width
        self.vx    = np.cos(theta)
        self.vy    = np.sin(theta)
        self.flux  = flux
        self.fCore = 1.0 - fWing
        self.fWing = fWing

        self.houghBinMax = 0

        
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
        self.insert(tmp, sigma=sigma, maskBit=satelliteBit)

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


    def insert(self, exposure, sigma=None, maskBit=None):
        """Plant this satellite trail in a given exposure.

        @param exposure       The exposure to plant in (accepts ExposureF, ImageF, MaskU or ndarray)
        @param sigma          The PSF size (as Gaussian sigma) to use.
        @param maskBit        Set pixels to this value.  (Don't plant a Double-Gaussian trail profile).

        This method serves a few purposes.

        (1) To search for a trail with profile similar to a PSF, we plant a PSF-shaped trail
            and measure its parameters for use in calibrating detection limits.
        (2) When we find a trail, our setMask() method calls this method with a maskBit to set.
        (3) For testing, we can insert trails.
        """
        
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
        w = (offset < hwidth)
        if maskBit:
            img[w] = maskBit
        else:
            A1  = 1.0/(2.0*np.pi*sigma**2)
            g1  = np.exp(-offset[w]**2/(2.0*sigma**2))
            A2  = 1.0/(2.0*np.pi*(2.0*sigma)**2)
            g2  = np.exp(-offset[w]**2/(2.0*(2.0*sigma)**2))
            img[w] += self.flux*(self.fCore*A1*g1 + self.fWing*A2*g2)
        
        return img

