#!/usr/bin/env python

import os
import time, datetime

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
    def __init__(self, nTotal, binMax, psfSigma):
        self.nTotal = nTotal
        self.binMax = binMax
        self.psfSigma = psfSigma

        
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
                sigma = satUtil.getExposurePsfSigma(exposure, minor=True)
                
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
                 skewLimit   = 40.0,
                 widthToPsfLimit = 0.4
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
        self.skewLimit         = skewLimit
        self.widthToPsfLimit   = widthToPsfLimit
        self.bRatio            = 1.0
        
    def getTrails(self, exposure, width=None, bins=None):

        if bins:
            #exp2 = type(exposure)(exposure, afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(600,2000)))
            #exposure = exp2
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
        
        psfsigma = satUtil.getExposurePsfSigma(exp, minor=True)
        print "PSF sigma: ", psfsigma
        self.sigmaSmooth = 2.0
        img = self._smooth(img, self.sigmaSmooth)
        _msk = exp.getMaskedImage().getMask()
        msk    = _msk.getArray()
        BAD    = _msk.getPlaneBitMask("BAD")
        CR     = _msk.getPlaneBitMask("CR")
        SAT    = _msk.getPlaneBitMask("SAT")
        INTRP  = _msk.getPlaneBitMask("INTRP")
        EDGE   = _msk.getPlaneBitMask("EDGE")
        SUSPECT= _msk.getPlaneBitMask("SUSPECT")
        MASK   = BAD | CR | SAT | INTRP | EDGE | SUSPECT
        
        xx, yy = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))

        print "getMoments"
        #   - get ellipticities and thetas
        self.sumI, self.center, self.center_perp, self.ellip, self.theta0, \
            self.ellipCal, self.thetaCal, \
            self.skew, self.skew_perp, self.b = self._getMoments(exp, width=width)

        print "... where ... "
        #   - cull unsuitable pixels
        faint_test = (
            ( np.abs(self.ellip - self.ellipCal) < self.eRange )
            & (img > self.luminosityLimit*self.noise) & (img < self.luminosityMax*self.noise) 
            & (np.abs(self.center_perp) < self.centerLimit)
            & (np.abs(self.center) < 2.0*self.centerLimit)
            & ~(msk & MASK)
            & (np.abs(self.skew_perp) < self.skewLimit)
            & (np.abs(self.skew) < 2.0*self.skewLimit)
            & (np.abs(self.b - self.bRatio) < self.widthToPsfLimit)
        )

        self.medium_factor = 2.0
        self.mediumLimit = 10.0*self.luminosityLimit #0.5
        self.mediumScale = 1.0
        medium_test = (
            ( np.abs(self.ellip - self.ellipCal/self.mediumScale) < self.medium_factor*self.eRange )
            & (img > self.mediumLimit*self.noise) & (img < self.luminosityMax*self.noise) 
            & (np.abs(self.center_perp) < self.centerLimit/self.medium_factor)
            & (np.abs(self.center) < 2.0*self.centerLimit/self.medium_factor)
            & ~(msk & MASK)
            & (np.abs(self.skew_perp) < self.skewLimit/self.medium_factor)
            & (np.abs(self.skew) < 2.0*self.skewLimit/self.medium_factor)
            & (np.abs(self.b - self.mediumScale*self.bRatio) < self.medium_factor*self.widthToPsfLimit)
        )
        
        self.bright_factor = 6.0
        self.brightLimit = 20.0*self.luminosityLimit #2.0
        self.brightScale = 1.0
        bright_test = (
            ( np.abs(self.ellip - self.ellipCal/self.brightScale) < self.bright_factor*self.eRange )
            & (img > self.brightLimit*self.noise) & (img < self.luminosityMax*self.noise) 
            & (np.abs(self.center_perp) < self.centerLimit/self.bright_factor)
            & (np.abs(self.center) < 2.0*self.centerLimit/self.bright_factor)
            & ~(msk & MASK)
            & (np.abs(self.skew_perp) < self.skewLimit/self.bright_factor)
            & (np.abs(self.skew) < 2.0*self.skewLimit/self.bright_factor)
            & (np.abs(self.b - self.brightScale*self.bRatio) < self.bright_factor*self.widthToPsfLimit)
        )

        self.wy, self.wx = np.where(faint_test | medium_test | bright_test)
        
        print "hesse"
        #   - convert suiltable pixels to hesse form (r,theta)
        self.r, self.theta = self._hesseForm(self.theta0[self.wy,self.wx],
                                             xx[self.wy,self.wx], yy[self.wy,self.wx])


        self.yy = yy[self.wy,self.wx]
        self.xx = xx[self.wy,self.wx]

        print "Hough"
        #   - bin and return detections
        self.rs, self.ts, self.xfin, self.yfin, binMax = self._houghTransform(self.r, self.theta,
                                                                              xx[self.wy,self.wx],
                                                                              yy[self.wy,self.wx])
        
        trails = SatelliteTrailList(len(self.r), max(binMax), psfsigma)
        for r,t,x,b in zip(self.rs, self.ts, self.xfin, binMax):
            print "Trail: ", bins*r, t, len(x), b
            trail = SatelliteTrail(bins*r, t)
            trail.nAboveThresh = len(x)
            trail.houghBinMax = b
            trails.append(trail)

            
        md = exp.getMetadata()
        v, c = 0, 0
        if 'VISIT' in md.paramNames():
            v, c = md.get('VISIT', 0), md.get('CCD_REGISTRY', 0)
        basedir = os.environ.get("SATELLITE_DATA", "/home/bick/sandbox/hough/data")
        path = os.path.join(basedir, "%04d" % (v))
        try:
            os.mkdir(path)
        except:
            pass
        print "plotting"
        self._debugPlot(img, trails, os.path.join(path,"satdebug-%05d-%03d-b%02d.png" % (v, c, self.bins)))

        return trails

        
    def _debugPlot(self, img, trails, pngfile):
        
        ###################################
        # a debug figure
        ###################################
        py, px = 2, 4
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
            fig.suptitle(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # pixel plot
            ax = fig.add_subplot(py, px, 1)
            ax.imshow(np.arcsinh(img), cmap="gray", origin='lower')
            ax.plot(self.xx, self.yy, 'r.', ms=1.0)
            ny, nx = img.shape
            for i,trail in enumerate(trails):
                x, y = trail.trace(nx, ny, offset=30, bins=self.bins)
                ax.plot(x, y, colors[i%4]+'-')
                x, y = trail.trace(nx, ny, offset=-30, bins=self.bins)
                ax.plot(x, y, colors[i%4]+'-')
            ax.set_xlim([0, nx])
            ax.set_ylim([0, ny])
            font(ax)
            
            # hough  r vs theta
            ax = fig.add_subplot(py, px, 2)
            ax.plot(self.theta, self.bins*self.r, 'k.', ms=1.0, alpha=0.5)
            for i,trail in enumerate(trails):
                ax.plot(trail.theta, trail.r, 'o', mfc='none', mec=colors[i%4], ms=10)
                ax.add_patch(Rectangle( (trail.theta - dt, trail.r - self.bins*dr),
                                        2*dt, 2*self.bins*dr, facecolor='none', edgecolor=colors[i%4]))
            ax.set_xlabel("Theta", size='small')
            ax.set_ylabel("r", size='small')
            ax.set_xlim([0.0, 2.0*np.pi])
            ax.set_ylim([0.0, self.bins*ny])
            ax.text(0.95, 0.95, "N=%d" % (len(self.theta)), size='xx-small',
                    horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
            font(ax)

            # e vs theta
            ax = fig.add_subplot(py, px, 3)
            stride = int(1.0*self.theta0.size/20000)
            if stride < 1:
                stride = 1
            #ax.plot(self.theta0[::stride], self.ellip[::stride], '.k', ms=0.2, alpha=0.2)
            ax.scatter(self.theta0[::stride], self.ellip[::stride],
                       c=np.clip(self.center[::stride], 0.0, 2.0*self.centerLimit),
                       s=0.4, alpha=0.4, edgecolor='none')
            ax.scatter(self.theta0[self.wy,self.wx], self.ellip[self.wy,self.wx],
                       c=np.clip(self.center[self.wy,self.wx], 0.0, 2.0*self.centerLimit),
                       s=0.8, alpha=0.8, edgecolor='none')
            ax.hlines([self.ellipCal], -np.pi/2.0, np.pi/2.0, color='m', linestyle='-')
            fhlines = [
                self.ellipCal - self.eRange,
                self.ellipCal + self.eRange,
            ]
            bhlines = [
                self.ellipCal/self.brightScale - self.bright_factor*self.eRange,
                self.ellipCal/self.brightScale + self.bright_factor*self.eRange,
            ]
            ax.hlines(fhlines, -np.pi/2.0, np.pi/2.0, color='m', linestyle='--')
            ax.hlines(bhlines, -np.pi/2.0, np.pi/2.0, color='c', linestyle='--')
            ax.set_xlabel("Theta", size='small')
            ax.set_ylabel("e", size='small')
            ax.set_xlim([-np.pi/2.0, np.pi/2.0])
            ax.set_ylim([0.0, 1.0])
            #ax.set_ylim([self.ellipCal-3.0*self.eRange, self.ellipCal+3.0*self.eRange])
            font(ax)

            # centroid vs flux
            ax = fig.add_subplot(py, px, 4)
            ax.scatter(self.center[::stride], img[::stride]/self.noise,
                       c=np.clip(self.center_perp[::stride], 0.0, 2.0*self.centerLimit), marker='.', s=1.0,
                       alpha=0.5, edgecolor='none')
            ax.scatter(self.center[self.wy,self.wx], img[self.wy,self.wx]/self.noise,
                       c=np.clip(self.center_perp[self.wy,self.wx], 0.0, 2.0*self.centerLimit),
                       s=2.0, alpha=1.0, edgecolor='none')
            ax.vlines([self.centerLimit, self.centerLimit/self.bright_factor], 0.001, 100, color='k', linestyle='--')
            ax.hlines([self.luminosityLimit], 0.01, 10, color='k', linestyle='--')
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Center", size='small')
            ax.set_ylabel("Flux", size='small')
            ax.set_xlim([0.01, 10])
            ax.set_ylim([0.001, 100])
            font(ax)


            # b versus skew
            ax = fig.add_subplot(py, px, 5)
            ax.scatter(self.skew, self.b,
                       c=np.clip(self.center, 0.0, 2.0*self.centerLimit), marker='.', s=2.0,
                       alpha=0.5, edgecolor='none')
            ax.scatter(self.skew[self.wy,self.wx], self.b[self.wy,self.wx],
                       c=np.clip(self.center[self.wy,self.wx], 0.0, 2.0*self.centerLimit), marker='.', s=4.0,
                       alpha=1.0, edgecolor='none')
            ax.vlines([self.skewLimit, self.skewLimit/self.bright_factor], 0, 3.0, linestyle='--', color='k')
            fhlines = [
                self.bRatio - self.widthToPsfLimit,
                self.bRatio + self.widthToPsfLimit,
            ]
            bhlines = [
                self.brightScale*self.bRatio - self.bright_factor*self.widthToPsfLimit,
                self.brightScale*self.bRatio + self.bright_factor*self.widthToPsfLimit,
            ]
            ax.hlines(fhlines, 0, 3.0*self.skewLimit, linestyle='--', color='m')
            ax.hlines(bhlines, 0, 3.0*self.skewLimit, linestyle='--', color='c')
            ax.set_xlabel("Skew", size='small')
            ax.set_ylabel("B", size='small')
            ax.set_xlim([0.0, 3.0*self.skewLimit])
            ax.set_ylim([0.0, 2.0])
            font(ax)

            
            for i,trail in enumerate(trails[0:px*py-5]):
                ax = fig.add_subplot(py,px,6+1*i)
                ax.plot(self.theta, self.bins*self.r, 'k.', ms=1.0, alpha=0.8)
                ax.plot(self.theta_new, self.bins*self.r_new, 'r.', ms=1.0, alpha=0.8)
                ax.plot(trail.theta, trail.r, 'o', mfc='none', ms=20, mec=colors[i%4])
                ax.set_xlabel("Theta", size='small')
                ax.set_ylabel("r", size='small')
                rmin, rmax = trail.r - dr, trail.r + dr
                tmin, tmax = trail.theta - dt, trail.theta + dt
                ax.set_xlim([tmin, tmax])
                ax.set_ylim([rmin, rmax])
                w, = np.where( (np.abs(self.theta - trail.theta) < dt) &
                               (np.abs(self.bins*self.r - trail.r) < dr))
                w_new, = np.where( (np.abs(self.theta_new - trail.theta) < dt) &
                                   (np.abs(self.bins*self.r_new - trail.r) < dr))
                ax.text(0.95, 0.95, "N=%d" % (len(w)), size='xx-small',
                        horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
                ax.text(0.95, 0.90, "N=%d" % (len(w_new)), size='xx-small',color = 'r',
                        horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
                font(ax)

            
            fig.savefig(pngfile)
            
        
    def _smooth(self, img, sigma):

        # if we're heavily binned, we're already smoothed
        if self.bins > 4:
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
            sigmaInsert = satUtil.getExposurePsfSigma(exposure, minor=True)
            sigmaSmooth = self.sigmaSmooth
            maskBit = None
            nSigma = 7.0
        else:
            sigmaInsert = width/2.0
            sigmaSmooth = self.sigmaSmooth
            maskBit = 1.0
            nSigma = 1.0
        calTrail.insert(cal, sigma=sigmaInsert/self.bins, maskBit=maskBit, nSigma=nSigma)
        # make sure we smooth in exactly the same way!
        cal = self._smooth(cal, sigmaSmooth)

        #fig = figure.Figure()
        #can = FigCanvas(fig)
        #ax = fig.add_subplot(111)
        #ax.imshow(cal)
        #fig.savefig("junk.png")
        
        ####################################
        # measure both the real moments and cal image
        ####################################
        convolved = satUtil.momentConvolve2d(img, self.kx, self.kernelSigma)
        sumI, ximg, yimg, xximg, yyimg, xyimg, x3img, y3img = convolved
        
        convolved_cal = satUtil.momentConvolve2d(cal, self.kx, self.kernelSigma)
        xcen, ycen = self.kernelSize//2, self.kernelSize//2
        _, xcal, ycal, xxcal, yycal, xycal, x3cal, y3cal = [c[ycen,xcen] for c in convolved_cal]

        center = np.sqrt(ximg**2 + yimg**2)
        ellip, theta, B       = satUtil.momentToEllipse(xximg, yyimg, xyimg)
        center_perp = np.abs(ximg*np.sin(theta) - yimg*np.cos(theta))
        skew = np.sqrt(x3img**2 + y3img**2)
        skew_perp = np.abs(x3img*np.sin(theta) - y3img*np.cos(theta))
        ellipCal, thetaCal, bCal = satUtil.momentToEllipse(xxcal, yycal, xycal)
        return sumI, center, center_perp, ellip, theta, ellipCal, thetaCal, skew, skew_perp, B/bCal
        

    def _hesseForm(self, theta, xx, yy):
        """ return r,theta """

        # convert to 0..2pi range
        w = np.where(theta < 0)
        theta_tmp0 = theta.copy()
        theta_tmp0[w] += 2.0*np.pi

        # convert to hesse theta.  Try adding pi/2 and correct if r < 0 (we define r > 0)
        theta_tmp0 = theta_tmp0 + np.pi/2.0
        theta_tmp = theta_tmp0.copy()

        r0   = xx*np.cos(theta_tmp) + yy*np.sin(theta_tmp)
        neg = np.where(r0 < 0.0)[0]
        theta_tmp[neg] += np.pi
        cycle = np.where(theta_tmp > 2.0*np.pi)[0]
        theta_tmp[cycle] -= 2.0*np.pi
        r0   = xx*np.cos(theta_tmp) + yy*np.sin(theta_tmp)
        
        return r0, theta_tmp
        

    def _houghTransform(self, r_in, theta_in, xx_in, yy_in):
        """ return list(SatelliteTrails) """

        # wrap theta~0 to above 2pi y     # wrap theta~2pi to near near
        thresh = 0.2
        w_0,   = np.where( theta_in < thresh )
        w_2pi, = np.where( 2.0*np.pi - theta_in < thresh )

        r0     = np.append(r_in,   r_in[w_0]) 
        theta0 = np.append(theta_in, theta_in[w_0] + 2.0*np.pi)
        xx0    = np.append(xx_in,  xx_in[w_0])
        yy0    = np.append(yy_in,  yy_in[w_0])
        
        r0     = np.append(r0,     r_in[w_2pi])    
        theta0 = np.append(theta0, theta_in[w_2pi] - 2.0*np.pi)
        xx0    = np.append(xx0,    xx_in[w_2pi])
        yy0    = np.append(yy0,    yy_in[w_2pi])

        
        # things get slow with more than ~1000 points, shuffle and cut
        points = len(r0)
        maxPoints = 1000

        r, theta, xx, yy = r0, theta0, xx0, yy0
        if points > maxPoints:
            idx = np.arange(points, dtype=int)
            np.random.shuffle(idx)
            idx = idx[:maxPoints]
            r, theta, xx, yy = r0[idx], theta0[idx], xx0[idx], yy0[idx]

            
        # improve the r,theta locations
        self.r_new, self.theta_new, _r, _xx, _yy = hesse.hesse_iter(theta, xx, yy, niter=3)

        r_max = 1.0
        if len(xx):
            r_max = np.sqrt(xx.max()**2 + yy.max()**2)
                
        # bin the data in r,theta space; get r,theta that pass our threshold as a satellite trail
        bin2d, r_edge, theta_edge, rs, thetas, idx = hesse.hesse_bin(self.r_new, self.theta_new,
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
        
