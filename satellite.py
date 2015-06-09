#!/usr/bin/env python

import time

import numpy as np
import numpy.fft as fft

import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as ellipses

import satellite_utils as satUtil
import hesse_cluster as hesse


class SatelliteTrailList(list):
    def __init__(self, nTotal, binMax):
        self.nTotal = nTotal
        self.binMax = binMax

        
class SatelliteTrail(object):
    def __init__(self, r, theta, flux=1.0, f_wing=0.1):
        self.r     = r
        self.theta = theta
        self.vx    = np.cos(theta)
        self.vy    = np.sin(theta)
        self.flux  = flux
        self.f_core = 1.0 - f_wing
        self.f_wing = f_wing

        self.nAboveThresh = 0
        self.houghBinMax = 0
        
    def setMask(self, exposure, nSigma=7.0):

        msk = exposure.getMaskedImage().getMask()
        sigma = satUtil.getExposurePsfSigma(exposure)
        satellitePlane = msk.addMaskPlane("SATELLITE")
        tmp = type(msk)(msk.getWidth(), msk.getHeight())
        self.insert(tmp, sigma=sigma, nSigma=nSigma, maskBit=satellitePlane)
        msk |= tmp
        # return the number of masked pixels
        return len(np.where(tmp.getArray() > 0)[0])
        
    def trace(self, nx, ny, offset=0):
        x = np.arange(nx)
        y = (self.r + offset - x*self.vx)/self.vy
        w, = np.where( (x > 0) & (x < nx) & (y > 0) & (y < ny) )
        return x[w], y[w]


    def insert(self, exposure, sigma=None, nSigma=7.0, maskBit=None):

        # Handle Exposure, Image, ndarray
        if isinstance(exposure, afwImage.ExposureF):
            img = exposure.getMaskedImage().getImage().getArray()
            nx, ny = exposure.getWidth(), exposure.getHeight()
            if sigma is None:
                sigma = satUtil.getExposurePsfSigma(exposure)
                
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

        # only bother updating the pixels within 5-sigma of the line
        wy,wx  = np.where(offset < nSigma*sigma)
        A1  = 1.0/(2.0*np.pi*sigma**2)
        g1  = np.exp(-offset[wy,wx]**2/(2.0*sigma**2))
        A2  = 1.0/(2.0*np.pi*(2.0*sigma)**2)
        g2  = np.exp(-offset[wy,wx]**2/(2.0*(2.0*sigma)**2))
        if maskBit:
            img[wy,wx] = maskBit
        else:
            img[wy,wx] += self.flux*(self.f_core*A1*g1 + self.f_wing*A2*g2)
        
        return img

        

class SatelliteFinder(object):

    def __init__(self,
                 kernelSigma = 15,
                 kernelSize  = 31,
                 centerLimit = 1.0,
                 eRange      = 0.1,
                 houghThresh = 20,
                 houghBins   = 256,
                 luminosityLimit = 4.0,
                 luminosityMax = 10.0,
             ):
        """ """
        
        self.kernelSigma       = kernelSigma
        
        self.kernelSize        = kernelSize
        self.kx = np.arange(kernelSize) - kernelSize//2
        self.ky = np.arange(kernelSize) - kernelSize//2

        self.centerLimit       = centerLimit
        self.eRange            = eRange
        self.houghThresh       = houghThresh
        self.houghBins         = houghBins
        self.luminosityLimit   = luminosityLimit
        self.luminosityMax     = luminosityMax
        
        self.preSmooth = False
        
        
    def getTrails(self, exposure, width=None):

        #   - smooth 
        img = exposure.getMaskedImage().getImage().getArray()
        noise = img.std()
        psfsigma = satUtil.getExposurePsfSigma(exposure)
        k = 2*int(6.0*psfsigma) + 1
        kk = np.arange(k) - k//2
        gauss = (1.0/np.sqrt(2.0*np.pi))*np.exp(-kk*kk/(2.0*psfsigma))
        img = satUtil.separableConvolve(img, gauss, gauss)

        
        xx, yy = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))
        
        #   - get ellipticities and thetas
        center, ellip, theta0, ellipCal, thetaCal = self._getMoments(exposure, width=width)
        
        #   - cull unsuitable pixels
        wy,wx = np.where(
            np.abs( (ellip - ellipCal) < self.eRange )
            & (img > self.luminosityLimit*noise) & (img < self.luminosityMax*noise) 
            & (np.abs(center) < self.centerLimit)
        )
        
        #   - convert suiltable pixels to hesse form (r,theta)
        r, theta = self._hesseForm(theta0[wy,wx], xx[wy,wx], yy[wy,wx])

        self.r = r
        self.theta = theta
        self.yy = yy[wy,wx]
        self.xx = xx[wy,wx]
        
        #   - bin and return detections
        rs, ts, xfin, yfin, binMax = self._houghTransform(r, theta, xx[wy,wx], yy[wy,wx])
        
        trails = SatelliteTrailList(len(r), max(binMax))
        for r,t,x,b in zip(rs, ts, xfin, binMax):
            trail = SatelliteTrail(r, t)
            trail.nAboveThresh = len(x)
            trail.houghBinMax = b
            trails.append(trail)


        ###################################
        # a debug figure
        ###################################
        debug = True
        if debug:

            fig = figure.Figure()
            can = FigCanvas(fig)
            ax = fig.add_subplot(221)
            ax.imshow(np.arcsinh(img), cmap="gray", origin='lower')
            for trail in trails:
                ny, nx = img.shape
                x, y = trail.trace(nx, ny, offset=20)
                ax.plot(x, y, 'r-')
                x, y = trail.trace(nx, ny, offset=-20)
                ax.plot(x, y, 'r-')

            ax = fig.add_subplot(222)
            ax.plot(self.theta, self.r, 'r.', markersize=2.0)
            for trail in trails:
                ax.plot(trail.theta, trail.r, 'go', mfc=None)
            ax.set_xlabel("Theta")
            ax.set_ylabel("r")
            ax = fig.add_subplot(223)
            ax.plot(theta0, ellip, '.k', ms=0.3, alpha=0.2)
            ax.set_xlabel("Theta")
            ax.set_ylabel("e")

            ax = fig.add_subplot(224)
            ax.loglog(center[wy,wx], img[wy,wx]/noise, '.k', ms=1.0, alpha=0.5)
            ax.set_xlabel("Center")
            ax.set_ylabel("Flux")
            
            md = exposure.getMetadata()
            v, c = md.get('VISIT', 0), md.get('CCD_REGISTRY', 0)
            fig.savefig("satdebug-%05d-%03d.png" % (v, c))
            
        return trails


        
    def _getMoments(self, exposure, width=None):
        """ return delta-centroid, ellip, theta """

        dx, dy = exposure.getWidth(), exposure.getHeight()
        img = exposure.getMaskedImage().getImage().getArray()

        ##################################
        # make a calibration trail
        ##################################
        calTrail = SatelliteTrail(self.kernelSize//2, 0.0)
        cal = np.zeros((self.kernelSize, self.kernelSize))
        if not width:
            sigma = satUtil.getExposurePsfSigma(exposure)
        else:
            sigma = width/2.0
        calTrail.insert(cal, sigma=sigma)

        
        convolved = satUtil.momentConvolve2d(img, self.kx, self.kernelSigma)
        ximg, yimg, xximg, yyimg, xyimg = convolved
        
        convolved_cal = satUtil.momentConvolve2d(cal, self.kx, self.kernelSigma)
        xcen, ycen = self.kernelSize//2, self.kernelSize//2
        xcal, ycal, xxcal, yycal, xycal = [c[ycen,xcen] for c in convolved_cal]

        center = np.sqrt(ximg**2 + yimg**2)
        ellip, theta       = satUtil.momentToEllipse(xximg, yyimg, xyimg)
        ellipCal, thetaCal = satUtil.momentToEllipse(xxcal, yycal, xycal)

        return center, ellip, theta, ellipCal, thetaCal
        

    def _hesseForm(self, theta, xx, yy):
        """ return r,theta """

        # convert to 0..2pi range
        w = np.where(theta < 0)
        theta_tmp0 = theta.copy()
        theta_tmp0[w] += 2.0*np.pi

        # convert to hesse theta.  Try adding pi/2 and correct if r < 0 (we define r > 0)
        theta_tmp0 = theta_tmp0 + np.pi/2.0
        theta_tmp = theta_tmp0.copy()

        r   = xx*np.cos(theta_tmp) + yy*np.sin(theta_tmp)
        neg = np.where(r < 0.0)[0]
        theta_tmp[neg] += np.pi
        cycle = np.where(theta_tmp > 2.0*np.pi)[0]
        theta_tmp[cycle] -= 2.0*np.pi
        r   = xx*np.cos(theta_tmp) + yy*np.sin(theta_tmp)
        return r, theta_tmp
        

    def _houghTransform(self, r_in, theta_in, xx_in, yy_in):
        """ return list(SatelliteTrails) """

        # things get slow with more than ~1000 points, shuffle and cut
        points = len(r_in)
        maxPoints = 1000
        r, theta, xx, yy = r_in, theta_in, xx_in, yy_in
        if points > maxPoints:
            idx = np.arange(points, dtype=int)
            np.random.shuffle(idx)
            idx = idx[:maxPoints]
            r, theta, xx, yy = r_in[idx], theta_in[idx], xx_in[idx], yy_in[idx]

        # improve the r,theta locations
        r_new, theta_new, _r, _xx, _yy = hesse.hesse_iter(theta, xx, yy, niter=0)

        # bin the data in r,theta space; get r,theta that pass our threshold as a satellite trail
        r_max = 1.0
        if len(xx):
            r_max = max(xx.max(), yy.max())
        bin2d, r_edge, theta_edge, rs, thetas, idx = hesse.hesse_bin(r_new, theta_new,
                                                                     bins=self.houghBins, r_max=r_max,
                                                                     ncut=self.houghThresh)
        
        numLocus = len(thetas)
        xfin, yfin, binMax = [], [], []
        for i in range(numLocus):
            xfin.append(xx[idx[i]])
            yfin.append(yy[idx[i]])
            binMax.append(len(idx[i]))
        if numLocus == 0:
            print "Max", bin2d.sum()
            binMax = [bin2d.max()]
        return rs, thetas, xfin, yfin, binMax
        
