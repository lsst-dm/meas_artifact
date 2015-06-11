#!/usr/bin/env python

import time

import numpy as np
import numpy.fft as fft

import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
from matplotlib.patches import Rectangle

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as ellipses
import lsst.afw.math as afwMath

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
        
    def trace(self, nx, ny, offset=0, bins=1):
        x = np.arange(nx)
        y = (self.r/bins + offset - x*self.vx)/self.vy
        w, = np.where( (x > 0) & (x < nx) & (y > 0) & (y < ny) )
        return x[w], y[w]


    def insert(self, exposure, sigma=None, nSigma=7.0, maskBit=None):

        if sigma < 1.0:
            sigma = 1.0
        
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
                 centerLimit = 0.8,
                 eRange      = 0.06,
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
        
        
    def getTrails(self, exposure, width=None, bins=None):

        if bins:
            exp = type(exposure)(afwMath.binImage(exposure.getMaskedImage(), bins))
            exp.setMetadata(exposure.getMetadata())
            exp.setPsf(exposure.getPsf())
        else:
            exp = exposure
            bins = 1
        self.bins = bins
            
        #   - smooth 
        img = exp.getMaskedImage().getImage().getArray()
        self.noise = img.std()
        
        psfsigma = satUtil.getExposurePsfSigma(exp)
        img = self._smooth(img, psfsigma)
        _msk = exp.getMaskedImage().getMask()
        msk = _msk.getArray()
        BAD = _msk.getPlaneBitMask("BAD")
        
        xx, yy = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))
        
        #   - get ellipticities and thetas
        self.sumI, self.center, self.ellip, self.theta0, \
            self.ellipCal, self.thetaCal = self._getMoments(exp, width=width)
        
        #   - cull unsuitable pixels
        self.wy,self.wx = np.where(
            np.abs( (self.ellip - self.ellipCal) < self.eRange )
            & (img > self.luminosityLimit*self.noise) & (img < self.luminosityMax*self.noise) 
            & (np.abs(self.center) < self.centerLimit)
            & ~(msk & BAD)
        )
        
        #   - convert suiltable pixels to hesse form (r,theta)
        self.r, self.theta = self._hesseForm(self.theta0[self.wy,self.wx],
                                             xx[self.wy,self.wx], yy[self.wy,self.wx])


        self.yy = yy[self.wy,self.wx]
        self.xx = xx[self.wy,self.wx]
        
        #   - bin and return detections
        rs, ts, xfin, yfin, binMax = self._houghTransform(self.r, self.theta,
                                                          xx[self.wy,self.wx], yy[self.wy,self.wx])
        
        trails = SatelliteTrailList(len(self.r), max(binMax))
        for r,t,x,b in zip(rs, ts, xfin, binMax):
            trail = SatelliteTrail(bins*r, t)
            trail.nAboveThresh = len(x)
            trail.houghBinMax = b
            trails.append(trail)

            
        md = exp.getMetadata()
        v, c = 0, 0
        if 'VISIT' in md.paramNames():
            v, c = md.get('VISIT', 0), md.get('CCD_REGISTRY', 0)
        self._debugPlot(img, trails, "satdebug-%05d-%03d-b%02d.png" % (v, c, self.bins))

        return trails

        
    def _debugPlot(self, img, trails, pngfile):
        
        ###################################
        # a debug figure
        ###################################
        debug = True
        if debug:

            dr = 100
            dt = 0.2
            colors = 'm', 'c', 'g', 'r'
            
            def font(ax):
                for t in ax.get_xticklabels() + ax.get_yticklabels():
                    t.set_size("xx-small")
            
            fig = figure.Figure()
            can = FigCanvas(fig)

            # pixel plot
            ax = fig.add_subplot(231)
            ax.imshow(np.arcsinh(img), cmap="gray", origin='lower')
            ny, nx = img.shape
            for i,trail in enumerate(trails):
                x, y = trail.trace(nx, ny, offset=20, bins=self.bins)
                ax.plot(x, y, colors[i%4]+'-')
                x, y = trail.trace(nx, ny, offset=-20, bins=self.bins)
                ax.plot(x, y, colors[i%4]+'-')
            ax.set_xlim([0, nx])
            ax.set_ylim([0, ny])
            font(ax)
            
            # hough  r vs theta
            ax = fig.add_subplot(232)
            ax.plot(self.theta, self.bins*self.r, 'k.', ms=1.0, alpha=0.5)
            for i,trail in enumerate(trails):
                ax.plot(trail.theta, trail.r, 'o', mfc='none', mec=colors[i%4], ms=10)
                ax.add_patch(Rectangle( (trail.theta - dt, trail.r - dr), 2*dt, 2*dr, facecolor='none', edgecolor=colors[i%4]))
            ax.set_xlabel("Theta", size='small')
            ax.set_ylabel("r", size='small')
            ax.text(0.95, 0.95, "N=%d" % (len(self.theta)), size='xx-small',
                    horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            font(ax)

            # e vs theta
            ax = fig.add_subplot(233)
            stride = int(len(self.theta0)/400)
            if stride < 1:
                stride = 1
            #ax.plot(self.theta0[::stride], self.ellip[::stride], '.k', ms=0.2, alpha=0.2)
            ax.scatter(self.theta0[::stride], self.ellip[::stride],
                       c=np.clip(self.center[::stride], 0.0, 2.0*self.centerLimit),
                       s=0.2, alpha=0.2, edgecolor='none')
            ax.hlines([self.ellipCal], -np.pi/2.0, np.pi/2.0, color='m', linestyle='-')
            ax.hlines([self.ellipCal - self.eRange], -np.pi/2.0, np.pi/2.0, color='m', linestyle='--')
            ax.hlines([self.ellipCal + self.eRange], -np.pi/2.0, np.pi/2.0, color='m', linestyle='--')
            ax.set_xlabel("Theta", size='small')
            ax.set_ylabel("e", size='small')
            ax.set_xlim([-np.pi/2.0, np.pi/2.0])
            ax.set_ylim([0.0, 1.0])
            font(ax)

            # centroid vs flux
            ax = fig.add_subplot(234)
            ax.loglog(self.center[::stride], img[::stride]/self.noise, '.k', ms=1.0, alpha=0.5)
            ax.set_xlabel("Center", size='small')
            ax.set_ylabel("Flux", size='small')
            ax.set_xlim([0.01, 10])
            ax.set_ylim([0.001, 100])
            font(ax)


            for i,trail in enumerate(trails[0:2]):
                ax = fig.add_subplot(2,3,5+i)
                ax.plot(self.theta, self.bins*self.r, 'k.', ms=1.0, alpha=0.8)
                ax.plot(trail.theta, trail.r, 'go', mfc='none', ms=20, mec=colors[i%4])
                ax.set_xlabel("Theta", size='small')
                ax.set_ylabel("r", size='small')
                rmin, rmax = trail.r - dr, trail.r + dr
                tmin, tmax = trail.theta - dt, trail.theta + dt
                ax.set_xlim([tmin, tmax])
                ax.set_ylim([rmin, rmax])
                w, = np.where( (np.abs(self.theta - trail.theta) < dt) &
                              (np.abs(self.bins*self.r - trail.r) < dr))
                ax.text(0.95, 0.95, "N=%d" % (len(w)), size='xx-small',
                        horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
                font(ax)
                
            
            fig.savefig(pngfile)
            
        
    def _smooth(self, img, sigma):

        # if we're heavily binned, we're already smoothed
        if self.bins > 2:
            return img
            
        k = 2*int(6.0*sigma) + 1
        kk = np.arange(k) - k//2
        gauss = (1.0/np.sqrt(2.0*np.pi))*np.exp(-kk*kk/(2.0*sigma))
        smth = satUtil.separableConvolve(img, gauss, gauss)
        return smth

        
    def _getMoments(self, exposure, width=None):
        """ return delta-centroid, ellip, theta """

        dx, dy = exposure.getWidth(), exposure.getHeight()
        img = exposure.getMaskedImage().getImage().getArray()

        ##################################
        # make a calibration trail
        ##################################
        calTrail = SatelliteTrail(self.kernelSize//2, 0)
        #calTrail = SatelliteTrail(0.0, np.pi/4.0)
        cal = np.zeros((self.kernelSize, self.kernelSize))
        if not width:
            sigma = satUtil.getExposurePsfSigma(exposure)
            maskBit = None
            nSigma = 7.0
        else:
            sigma = width/2.0
            maskBit = 1.0
            nSigma = 1.0
        calTrail.insert(cal, sigma=sigma/self.bins, maskBit=maskBit, nSigma=nSigma)
        # make sure we smooth in exactly the same way!
        cal = self._smooth(cal, sigma)

        fig = figure.Figure()
        can = FigCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(cal)
        fig.savefig("junk.png")
        
        ####################################
        # measure both the real moments and cal image
        ####################################
        convolved = satUtil.momentConvolve2d(img, self.kx, self.kernelSigma)
        sumI, ximg, yimg, xximg, yyimg, xyimg = convolved
        
        convolved_cal = satUtil.momentConvolve2d(cal, self.kx, self.kernelSigma)
        xcen, ycen = self.kernelSize//2, self.kernelSize//2
        _, xcal, ycal, xxcal, yycal, xycal = [c[ycen,xcen] for c in convolved_cal]

        center = np.sqrt(ximg**2 + yimg**2)
        ellip, theta       = satUtil.momentToEllipse(xximg, yyimg, xyimg)
        ellipCal, thetaCal = satUtil.momentToEllipse(xxcal, yycal, xycal)

        return sumI, center, ellip, theta, ellipCal, thetaCal
        

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
            binMax = [bin2d.max()]
        return rs, thetas, xfin, yfin, binMax
        
