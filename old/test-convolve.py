#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import satellite_utils as satUtil
import satellite as satell
import time

def main():

    nx, ny = 1024, 1024
    #nx, ny = 4*1024, 4*1024
    nx, ny = 256, 256
    #nx, ny = 128,128
    n = 4
    img = np.zeros((nx, ny))

    x = np.random.uniform(nx, size=n).astype(int)
    y = np.random.uniform(ny, size=n).astype(int)
    
    img[y, x] += 1000.0

    img += 10.0*np.random.normal(size=(nx, ny))
    
    sigma = 4.0
    k = 21
    kk = np.arange(k) - k//2
    kern = np.exp( -(kk)**2/(2.0*sigma**2) )

    t1 = time.time()
    for i in range(1):
        conv = satUtil.separableConvolve(img, kern, kern)
    t2 = time.time()
    for i in range(1):
        conv2 = satUtil.fftConvolve2d(img, [np.outer(kern,kern)])
    t3 = time.time()

    moments = satUtil.momentConvolve2d(img, kern, sigma)
    
    print t2-t1, t3-t2
    
    print img.shape, conv.shape, conv2[0].shape

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.imshow(img, cmap='gray', interpolation='none')
    ax = fig.add_subplot(312)
    ax.imshow(conv, cmap='gray', interpolation='none')
    ax = fig.add_subplot(313)
    ax.imshow(conv2[0], cmap='gray', interpolation='none')
    fig.savefig("conv.png")


    kernelSize = 31
    kx = np.arange(kernelSize) - kernelSize // 2
    calTrail = satell.SatelliteTrail(kernelSize//2, 0.0)
    cal = np.zeros((kernelSize, kernelSize))
    calTrail.insert(cal, sigma=sigma)
    
    convolved_cal = satUtil.momentConvolve2d(cal, kx, sigma)
    xcen, ycen = kernelSize//2, kernelSize//2
    xcal, ycal, xxcal, yycal, xycal = [c[ycen,xcen] for c in convolved_cal]
    ellipCal, thetaCal = satUtil.momentToEllipse(xxcal, yycal, xycal)
    print ellipCal, thetaCal


    show = list(moments)
    show.append(cal)
    fig = plt.figure()
    for i, im in enumerate(show):
        ax = fig.add_subplot(2, len(show)/2, 1+i)
        ax.imshow(im)
    fig.savefig("moment.png")


    

if __name__ == "__main__":
    main()
