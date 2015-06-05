#!/usr/bin/env python

import time
import numpy as np

import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg

import satellite as satell


def main():

    #####################################
    # create input traces
    #####################################
    print "Adding trails"
    
    noise  = 5.0
    #nx, ny = 512, 512
    nx, ny = 1024, 1024
    #nx, ny = 2048, 2048
    nx, ny = 2048, 4096
    kx, ky = 21, 21
    seeing = 4.0
    mimg = afwImage.MaskedImageF(nx, ny)
    psf = measAlg.DoubleGaussianPsf(kx, ky, seeing/2.35)
    exp = afwImage.ExposureF(mimg) #, psf)
    exp.setPsf(psf)
    nim = mimg.getImage().getArray()
    
    r_in     = [200,       400]
    theta_in = [np.pi/3.0, np.pi/7.0]
    flux_in  = [300.0,      400.0]
    trails_in = []
    for r,t,f in zip(r_in, theta_in, flux_in):
        trail = satell.SatelliteTrail(r, t, f)
        trail.insert(exp)
        trails_in.append(trail)

    nim += noise*np.random.normal(size=(ny, nx))

    #######################################
    # try to find them
    #######################################

    print "Trying to find them"
    t = time.time()
    kernelSigma = 9    # pixels
    kernelSize  = 31   # pixels
    centerLimit = 1.0   # about 1 pixel
    eRange      = 0.08   # about +/- 0.1
    
    houghThresh     = 40    # counts in a r,theta bins
    houghBins       = 256   # number of r,theta bins (i.e. 256x256)
    luminosityLimit = 4.0   # low cut on pixel flux
    
    finder = satell.SatelliteFinder(
        kernelSigma=kernelSigma,
        kernelSize=kernelSize,
        centerLimit=centerLimit,
        eRange=eRange,
        houghThresh=houghThresh,
        houghBins=houghBins,
        luminosityLimit=luminosityLimit
    )

    satelliteTrails = finder.getTrails(exp)

    print "Time: ", time.time() - t
    
    print "All done, plotting."
    
    for trail in trails_in:
        print trail.r, trail.theta
        
    for trail in satelliteTrails:
        #trail.setMask(exposure, maskPlane)
        print trail.r, trail.theta
        

    fig = figure.Figure()
    can = FigCanvas(fig)
    ax = fig.add_subplot(211)
    ax.imshow(nim, cmap="gray", origin='lower')
    for trail in satelliteTrails:
        x, y = trail.trace(*nim.shape, offset=10)
        ax.plot(x, y, 'r-')
        x, y = trail.trace(*nim.shape, offset=-10)
        ax.plot(x, y, 'r-')

    r, t = finder.r, finder.theta
    ax = fig.add_subplot(212)
    ax.plot(t, r, 'r.', markersize=2.0)
    for trail in satelliteTrails:
        ax.plot(trail.theta, trail.r, 'go')
    fig.savefig("input.png")

if __name__ == '__main__':
    main()
    
