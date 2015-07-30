import numpy as np
import numpy.fft as fft

import lssttools.functions as func
from lssttools import moment
from lssttools import utils as util

def fftSmooth2d(data, ixx, iyy, ixy):

    sx, sy = ixx**0.5, iyy**0.5
    kx, ky = [int(x)+1 if int(x) % 2 == 1 else int(x) for x in [5.0*sx, 5.0*sy]]

    ny, nx = data.shape
    kernel = np.zeros(data.shape)

    #egauss = func.EllipticalGauss(moment.Moment(ixx, iyy, ixy))
    egauss = func.DoubleGauss(np.sqrt(ixx))
    kernel = egauss.getImage(kx, kx/2, ky/2)

    kimg  = np.zeros(data.shape)
    kimg[ny/2-ky/2:ny/2+ky/2,nx/2-kx/2:nx/2+kx/2] += kernel

    #util.writeFits(kimg, "kimg.fits")
    kimgF = fft.fft2(fft.fftshift(kimg))
    dataF = fft.fft2(data)
    prodF = kimgF * dataF
    smth  = fft.ifft2(prodF).real

    return smth

def fftConvolve(data, kernel):

    ky, kx = kernel.shape
    ny, nx = data.shape
    kimg  = np.zeros(data.shape)
    kimg[ny/2-ky/2:ny/2+ky/2+1,nx/2-kx/2:nx/2+kx/2+1] += kernel

    #util.writeFits(kimg, "kimg.fits")
    kimgF = fft.fft2(fft.fftshift(kimg))
    dataF = fft.fft2(data)
    prodF = kimgF * dataF
    smth  = fft.ifft2(prodF).real

    return smth
