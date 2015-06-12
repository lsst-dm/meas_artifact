#!/usr/bin/env python

import numpy as np
import numpy.fft as fft
import scipy.signal as signal
import scipy.ndimage.filters as filt

import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as ellipses

def getExposurePsfSigma(exposure, minor=False):
    nx, ny = exposure.getWidth(), exposure.getHeight()
    midpixel = afwGeom.Point2D(nx//2, ny//2)
    psfshape = exposure.getPsf().computeShape(midpixel)
    axes  = ellipses.Axes(psfshape)
    if minor:
        return axes.getB()
    else:
        return np.sqrt(axes.getA()*axes.getB())

    
def fftConvolve2d(data, kernels):

    ny, nx = data.shape
    dataF = fft.fft2(data)

    convolved = []
    for kernel in kernels:
        ky, kx = kernel.shape
        kimg  = np.zeros(data.shape)
        kimg[ny/2-ky/2:ny/2+ky/2+1,nx/2-kx/2:nx/2+kx/2+1] += kernel

        kimgF = fft.fft2(fft.fftshift(kimg))
        prodF = kimgF * dataF
        
        conv  = fft.ifft2(prodF).real
        convolved.append(conv)
        
    return convolved


def separableConvolve(data, vx, vy):
    mode = 'reflect'
    out0 = filt.correlate1d(data, vx, mode=mode)
    out  = filt.correlate1d(out0, vy, mode=mode, axis=0)
    return out

    
def momentConvolve2d(data, k, sigma):

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
    
    return sumI, ix, iy, ixx, iyy, ixy, ix3, iy3

    
    
def momentToEllipse(ixx, iyy, ixy, lo_clip=1.0):

    tmp   = 0.5*(ixx + iyy)
    diff  = ixx - iyy
    tmp2  = np.sqrt(0.25*diff**2 + ixy**2)
    a2    = np.clip(tmp + tmp2, lo_clip, None)
    b2    = np.clip(tmp - tmp2, lo_clip, None)
    ellip = 1.0 - np.sqrt(b2/a2)
    theta = 0.5*np.arctan2(2.0*ixy, diff)

    return ellip, theta, np.sqrt(b2)


