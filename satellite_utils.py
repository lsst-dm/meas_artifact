#!/usr/bin/env python

import numpy as np
import numpy.fft as fft

import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as ellipses

def getExposurePsfSigma(exposure):
    nx, ny = exposure.getWidth(), exposure.getHeight()
    midpixel = afwGeom.Point2D(nx//2, ny//2)
    psfshape = exposure.getPsf().computeShape(midpixel)
    axes  = ellipses.Axes(psfshape)
    sigma = np.sqrt(axes.getA()*axes.getB())
    return sigma

    
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

    
def momentToEllipse(ixx, iyy, ixy, lo_clip=1.0):

    tmp   = 0.5*(ixx + iyy)
    diff  = ixx - iyy
    tmp2  = np.sqrt(0.25*diff**2 + ixy**2)
    a2    = np.clip(tmp + tmp2, lo_clip, None)
    b2    = np.clip(tmp - tmp2, lo_clip, None)
    ellip = 1.0 - np.sqrt(b2/a2)
    theta = 0.5*np.arctan2(2.0*ixy, diff)

    return ellip, theta


