#!/usr/bin/env python

import collections
import numpy                  as np
import scipy.ndimage.filters  as filt

import lsst.afw.geom          as afwGeom
import lsst.afw.geom.ellipses as ellipses


def getExposurePsfSigma(exposure, minor=False):
    """Helper function to extract the PSF size from an afwImage.Exposure object

    @param exposure    The exposure you want a PSF size for
    @param minor       Return the minor axis size of the PSF (default will be sqrt(a*b)).
    """
    nx, ny = exposure.getWidth(), exposure.getHeight()
    midpixel = afwGeom.Point2D(nx//2, ny//2)
    psfshape = exposure.getPsf().computeShape(midpixel)
    
    axes  = ellipses.Axes(psfshape)
    if minor:
        return axes.getB()
    else:
        return np.sqrt(axes.getA()*axes.getB())


def separableConvolve(data, vx, vy):
    """Convolve 2D data with 1D kernels in X and then Y

    @param data    The input 2D ndarray
    @param vx      The x-vector for convolution
    @param vy      The y-vector for convolution

    @return out    The convolved 2D array.
    """
    
    mode = 'reflect'
    out0 = filt.correlate1d(data, vx, mode=mode)
    out  = filt.correlate1d(out0, vy, mode=mode, axis=0)
    return out


def smooth(img, sigma):
    """Gaussian mmooth an image.

    @param img     The image to smooth
    @param sigma   The 'sigma' of the smoothing Gaussian

    @return smth   The smoothed image
    """
    k     = 2*int(6.0*sigma) + 1
    kk    = np.arange(k) - k//2
    gauss = (1.0/np.sqrt(2.0*np.pi))*np.exp(-kk*kk/(2.0*sigma))
    smth  = separableConvolve(img, gauss, gauss)
    return smth



# No docstring for a namedtuple (make it a class?)
# It's just a return value for the momentConvolve2d() function below
ImageMoment = collections.namedtuple("ImageMoment", "i0 ix iy ixx iyy ixy ixxx iyyy")


def momentConvolve2d(data, k, sigma, middleOnly=False):
    """Convolve an image with coefficient kernels to compute local 'moments'

    @param data       The input image
    @param k          A vector of indices (e.g. -3,-2,-1,0,1,2,3 )
    @param sigma      Gaussian sigma for an overall smoothing to avoid blowing up higher-order moments
    @param middleOnly Boolean to return the central pixel only (used for calibration images)

    return ImageMoment  A container with attributes for each moment: .i0 .ix .iy .ixx .iyy .ixy etc.

    Each of these convolutions uses a separable kernel, and many share a common convolution
    in at least one dimension.
    """
    
    # moments are  e.g.   sum(I*x) / sum(I)
    
    gauss = np.exp(-k**2/(2.0*sigma**2))
    
    kk = k*k
    k3 = kk*k
    k4 = kk*kk
    
    mode = 'reflect'

    # start with convolutions with our Gaussian in separately in X and Y
    gaussX = filt.correlate1d(data, gauss, mode=mode)
    gaussY = filt.correlate1d(data, gauss, mode=mode, axis=0)

    # zeroth order moment (i.e. a sum), convolve the X gaussian along Y
    sumI = filt.correlate1d(gaussX, gauss, mode=mode, axis=0)
    sumI[np.where(sumI == 0)] = 1.0e-7

    # normalize up front
    gaussX /= sumI
    gaussY /= sumI
    
    # Now use gaussX and gaussY to get the moments
    ix   = filt.correlate1d(gaussY, gauss*k, mode=mode)
    iy   = filt.correlate1d(gaussX, gauss*k, mode=mode, axis=0)
    ixx  = filt.correlate1d(gaussY, gauss*kk, mode=mode)
    iyy  = filt.correlate1d(gaussX, gauss*kk, mode=mode, axis=0)

    # cross term requires special attention.  Start from scratch.
    ixy0 = filt.correlate1d(data, gauss*k, mode=mode)
    ixy  = filt.correlate1d(ixy0, gauss*k, mode=mode, axis=0) /sumI

    # don't bother with 3rd order cross terms
    ix3  = filt.correlate1d(gaussY, gauss*k3, mode=mode)
    iy3  = filt.correlate1d(gaussX, gauss*k3, mode=mode, axis=0)

    values = sumI, ix, iy, ixx, iyy, ixy, ix3, iy3
    if middleOnly:
        ny, nx = data.shape
        values = [ x[ny//2,nx//2] for x in values ]
    return ImageMoment(*values)

    
    
