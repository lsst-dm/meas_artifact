#!/usr/bin/env python

import numpy as np
import numpy.fft as fft

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as ellipses

import satellite_utils as satUtil
import hesse_cluster as hesse

class SatelliteTrail(object):
    def __init__(self, r, theta, flux=1.0, f_wing=0.1):
        self.r     = r
        self.theta = theta
        self.vx    = np.cos(theta)
        self.vy    = np.sin(theta)
        self.flux  = flux
        self.f_core = 1.0 - f_wing
        self.f_wing = f_wing

        
    def setMask(self, exposure, nPsfWidth=3.0):
        pass

    def trace(self, nx, ny, offset=0):
        x = np.arange(nx)
        y = (self.r + offset - x*self.vx)/self.vy
        w, = np.where( (x > 0) & (x < nx) & (y > 0) & (y < ny) )
        return x[w], y[w]
        
    def insert(self, exposure, sigma=None):

        # Handle Exposure, Image, ndarray
        if isinstance(exposure, afwImage.ExposureF):
            img = exposure.getMaskedImage().getImage().getArray()
            nx, ny = exposure.getWidth(), exposure.getHeight()
            if sigma is None:
                sigma = satUtil.getExposurePsfSigma(exposure)
                
        elif isinstance(exposure, afwImage.ImageF):
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
        x = np.arange(nx).astype(int)
        y = np.arange(ny).astype(int)
            
        yy, xx = np.meshgrid(x, y)

        # plant the trail using the distance from our line
        # as the parameter in a 1D DoubleGaussian
        dot    = xx*self.vx + yy*self.vy
        offset = np.abs(dot - self.r)

        # only bother updating the pixels within 5-sigma of the line
        wy,wx  = np.where(offset < 5.0*sigma)
        A1  = 1.0/(2.0*np.pi*sigma**2)
        g1  = np.exp(-offset[wy,wx]**2/(2.0*sigma**2))
        A2  = 1.0/(2.0*np.pi*(2.0*sigma)**2)
        g2  = np.exp(-offset[wy,wx]**2/(2.0*(2.0*sigma)**2))
        img[wy,wx] += self.flux*(self.f_core*g1 + self.f_wing*g2)
        
        return img

        

class SatelliteFinder(object):

    def __init__(self,
                 kernelSigma = 9,
                 kernelSize  = 31,
                 centerLimit = 1.0,
                 eRange      = 0.1,
                 houghThresh = 20,
                 houghBins   = 256,
                 luminosityLimit = 0.2,
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
        
        self.preSmooth = False
        
        
    def getTrails(self, exposure):

        #   - smooth and do a luminosity cut
        img = exposure.getMaskedImage().getImage().getArray()
        noise = img.std()
        
        xx, yy = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))
        
        #   - get ellipticities and thetas
        center, ellip, theta, ellipCal, thetaCal = self._getMoments(exposure)
        
        #   - cull unsuitable pixels
        wy,wx = np.where(
            np.abs( (ellip - ellipCal) < self.eRange )
            & (img > self.luminosityLimit*noise)
            & (np.abs(center) < self.centerLimit)
        )
        
        #   - convert suiltable pixels to hesse form (r,theta)
        r, theta = self._hesseForm(theta[wy,wx], xx[wy,wx], yy[wy,wx])

        
        #   - bin and return detections
        rs, ts, xfin, yfin = self._houghTransform(r, theta, xx[wy,wx], yy[wy,wx])

        trails = []
        for r,t in zip(rs, ts):
            trails.append(SatelliteTrail(r, t))
        
        return trails

        
    def _smooth(self, exposure):
        pass
        
    def _getMoments(self, exposure):
        """ return delta-centroid, ellip, theta """

        dx, dy = exposure.getWidth(), exposure.getHeight()
        img = exposure.getMaskedImage().getImage().getArray()
        
        #############################
        # build the kernel
        #############################
        print "building the kernel"
        kxx, kyy = np.meshgrid(self.kx, self.ky)
        gauss = np.exp(-(kxx**2 + kyy**2)/(2.0*self.kernelSigma**2))
        
        xxkern = gauss*kxx**2
        xkern  = gauss*kxx
        yykern = gauss*kyy**2
        ykern  = gauss*kyy
        xykern = gauss*kxx*kyy

        #############################
        # convolve
        #############################
        print "convolving"
        kernels   = gauss, xkern, ykern, xxkern, yykern, xykern
        convolved = satUtil.fftConvolve2d(img, kernels)
        norm, ximg, yimg, xximg, yyimg, xyimg = convolved

        print "normalizing"
        # overall image intensity will creep in through gaussian, so normalize against it
        for tmp in ximg, yimg, xximg, yyimg, xyimg:
            tmp /= norm

            
        ##################################
        # make a calibration trail
        ##################################
        print "calibration trail"
        calTrail = SatelliteTrail(self.kx//2, 0.0)
        cal = np.zeros((self.kernelSize, self.kernelSize))
        calTrail.insert(cal, sigma=satUtil.getExposurePsfSigma(exposure))

        # we need only one point, so just do the product and sum (i.e. no convolution)
        norm_cal = (cal*gauss).sum()
        xxcal = (xxkern*cal).sum()/norm_cal
        xcal  = (xkern*cal).sum()/norm_cal
        yycal = (yykern*cal).sum()/norm_cal
        ycal  = (ykern*cal).sum()/norm_cal
        xycal = (xykern*cal).sum()/norm_cal

        center = np.sqrt(ximg**2 + yimg**2)
        ellip, theta       = satUtil.momentToEllipse(xximg, yyimg, xyimg)
        ellipCal, thetaCal = satUtil.momentToEllipse(xxcal, yycal, xycal)

        return center, ellip, theta, ellipCal, thetaCal


    def _hesseForm(self, theta, xx, yy):
        """ return r,theta """
        
        print "hesse", len(theta)
        theta_tmp0 = theta + np.pi/2.0
        theta_tmp = theta_tmp0.copy()
        
        r   = xx*np.cos(theta_tmp) + yy*np.sin(theta_tmp)
        neg = np.where(r < 0.0)[0]
        theta_tmp[neg] += np.pi
        cycle = np.where(theta_tmp > 2.0*np.pi)[0]
        theta_tmp[cycle] -= 2.0*np.pi
        r   = xx*np.cos(theta_tmp) + yy*np.sin(theta_tmp)
        return r, theta_tmp
        

        
    def _houghTransform(self, r, theta, xx, yy):
        """ return list(SatelliteTrails) """

        print "Hough", len(r)
        r_max = max(xx.max(), yy.max())
        r_new, t_new, _r, _xx, _yy = hesse.hesse_iter(theta, xx, yy, niter=2)
        bin2d, r_edge, t_edge, rs, ts, idx = hesse.hesse_bin(r_new, t_new,
                                                             bins=self.houghBins, r_max=r_max,
                                                             ncut=self.houghThresh)
        
        numLocus = len(ts)
        xfin = []
        yfin = []
        for i in range(numLocus):
            print "Locus ", i, ts[i], rs[i], len(idx[i])
            xfin.append(xx[idx[i]])
            yfin.append(yy[idx[i]])
            
        return rs, ts, xfin, yfin
