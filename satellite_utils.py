#!/usr/bin/env python

import numpy as np
import scipy.ndimage.filters as filt

import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as ellipses

import collections

def getExposurePsfSigma(exposure, minor=False):
    nx, ny = exposure.getWidth(), exposure.getHeight()
    midpixel = afwGeom.Point2D(nx//2, ny//2)
    psfshape = exposure.getPsf().computeShape(midpixel)
    
    axes  = ellipses.Axes(psfshape)
    if minor:
        return axes.getB()
    else:
        return np.sqrt(axes.getA()*axes.getB())


def separableConvolve(data, vx, vy):
    mode = 'reflect'
    out0 = filt.correlate1d(data, vx, mode=mode)
    out  = filt.correlate1d(out0, vy, mode=mode, axis=0)
    return out


def smooth(img, sigma):
    
    k     = 2*int(6.0*sigma) + 1
    kk    = np.arange(k) - k//2
    gauss = (1.0/np.sqrt(2.0*np.pi))*np.exp(-kk*kk/(2.0*sigma))
    smth  = separableConvolve(img, gauss, gauss)
    return smth

    
ImageMoment = collections.namedtuple("ImageMoment", "i0 ix iy ixx iyy ixy ixxx iyyy")

def momentConvolve2d(data, k, sigma, middleOnly=False):

    # moments are  e.g.   sum(I*x) / sum(I)
    
    gauss = np.exp(-k**2/(2.0*sigma**2))
    
    kk = k*k
    k3 = kk*k
    k4 = kk*kk
    
    mode = 'reflect'
    gaussX = filt.correlate1d(data, gauss, mode=mode)
    gaussY = filt.correlate1d(data, gauss, mode=mode, axis=0)
    
    sumI = filt.correlate1d(gaussX, gauss, mode=mode, axis=0)
    sumI[np.where(sumI == 0)] = 1.0e-7
    
    ix   = filt.correlate1d(gaussY, gauss*k, mode=mode) /sumI
    iy   = filt.correlate1d(gaussX, gauss*k, mode=mode, axis=0) /sumI
    ixx  = filt.correlate1d(gaussY, gauss*kk, mode=mode) /sumI
    iyy  = filt.correlate1d(gaussX, gauss*kk, mode=mode, axis=0) /sumI
    ixy0 = filt.correlate1d(data, gauss*k, mode=mode)
    ixy  = filt.correlate1d(ixy0, gauss*k, mode=mode, axis=0) /sumI
    
    ix3  = filt.correlate1d(gaussY, gauss*k3, mode=mode) /sumI
    iy3  = filt.correlate1d(gaussX, gauss*k3, mode=mode, axis=0) /sumI

    values = sumI, ix, iy, ixx, iyy, ixy, ix3, iy3
    if middleOnly:
        ny, nx = data.shape
        values = [ x[ny//2,nx//2] for x in values ]
    return ImageMoment(*values)

    
    
