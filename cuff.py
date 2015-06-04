#!/usr/bin/env python

import sys
import argparse
import itertools
import numpy as np
from scipy import ndimage as ndimg

#import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
from matplotlib.patches import Rectangle

import lsst.daf.persistence  as dafPersist
import lsst.afw.image        as afwImage
import lsst.afw.display.ds9  as ds9
import lsst.afw.geom         as afwGeom
import lsst.afw.geom.ellipses as ellipse

import lssttools.functions as func
import lssttools.moment as moment
import filter as filt
import line
import addTrace
import hesse_cluster as hesse

# - get intersections

# separate into a callable routine
# - take an exposure, add sat-trail to mask, return list of SatelliteTrail objects (theta, r, strength?)
#   - get moments
#   - smooth and do a luminosity cut
#   - get a list of satisfying xx,yy
#   - get the list of r,theta
#   - bin and return detections

# - write a MaskSatelliteTrailsTask
# - test on real exposure

def main(rootDir, visit, ccd, frame=1, title="", scale="zscale", zoom="to fit", trans=60):

    softwidth = 9
    dx = 31
    dy = 31
    feps = 0.1  # this is never quite what's expected
    bins = 256 #100,100 #200, 200 #128,128 #64,64 #32,32 #128,128
    navg = 0.2
    ncut = 40
    lumcut = 0.02
    centcut = 1.0
    gradcut = 10.0
    tiles = 1
    segment = 0.01
    lapcut = 1.0
    cell_sep = 5
    
    # make a butler and specify your dataId
    butler = dafPersist.Butler(rootDir)
    dataId = {'visit': visit, 'ccd':ccd}

    # get the exposure from the butler
    exposure = butler.get('calexp', dataId)
    midpixel = afwGeom.Point2D(1000, 2000)
    psfshape = exposure.getPsf().computeShape(midpixel)
    psfwidth = ellipse.Axes(psfshape).getA()
    print "PSF: ", psfwidth
    
    print "Loading"
    img0 = exposure.getMaskedImage().getImage().getArray()[:,:]

    #####################################
    # Add fakes
    ######################################

    print "Adding fakes"
    r_in = [500.0, 1000.0, 500.0, 2000.0, 100.0]
    t_in = [0.28, 0.28, 0.28+np.pi/2.0, 0.28, 0.28+1.0*np.pi/2.0]
    flux = [40.0, 60.0, 80.0, 100.0, 20.0]
    imgx = img0
    for i in range(len(r_in)):
        imgx = addTrace.addTrace(imgx, r_in[i], t_in[i], flux[i], sigma=psfwidth)

    fakex = addTrace.addTrace(np.zeros((dx,dy)), dx//2+1, 0.0, 1.0, sigma=psfwidth)
        
    #####################################
    # smooth
    #####################################
    print "smooth"
    kwid = 1.0
    img = filt.fftSmooth2d(imgx, kwid, kwid, 0.0)
    fake = filt.fftSmooth2d(fakex, kwid, kwid, 0.0)
    
    xx0, yy0 = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))

    sigma = img.std()

    #####################################
    # get the moments
    #####################################
    print "getting moments"
    softgauss = func.EllipticalGauss(moment.Moment(softwidth**2, softwidth**2, 0.0))
    soft = softgauss.getImage(dx, dy, dx/2, dy/2)
    
    xk, yk = np.meshgrid(np.arange(dx) - dx//2, np.arange(dy) - dy//2)
    xxkern = soft*xk**2
    xkern  = soft*xk
    yykern = soft*yk**2
    ykern  = soft*yk
    xykern = soft*xk*yk

    norm = filt.fftConvolve(img, soft)
    xximg = filt.fftConvolve(img, xxkern)/norm
    ximg  = filt.fftConvolve(img, xkern)/norm
    yyimg = filt.fftConvolve(img, yykern)/norm
    yimg  = filt.fftConvolve(img, ykern)/norm
    xyimg = filt.fftConvolve(img, xykern)/norm

    norm_sum = (fake*soft).sum()
    xxf = (xxkern*fake).sum()/norm_sum
    xf  = (xkern*fake).sum()/norm_sum
    yyf = (yykern*fake).sum()/norm_sum
    yf  = (ykern*fake).sum()/norm_sum
    xyf = (xykern*fake).sum()/norm_sum
    
    #####################################
    # convert to A,B,theta
    #####################################
    print "convert ellipses"
    def toEllipse(ixx, iyy, ixy, lo_clip=1.0):
        tmp = 0.5*(ixx + iyy)
        diff = ixx - iyy
        tmp2 = np.sqrt(0.25*diff**2 + ixy**2)
        a2 = np.clip(tmp + tmp2, lo_clip, None)
        b2 = np.clip(tmp - tmp2, lo_clip, None)
        ellip = 1.0 - np.sqrt(b2/a2)
        theta = 0.5*np.arctan2(2.0*ixy, diff)
        return ellip, theta
    def gradient(img):
        img_dx = np.zeros(img.shape)
        img_dx[:,1:-1] = 0.5*(img[:,2:] - img[:,:-2])
        img_dy = np.zeros(img.shape)
        img_dy[1:-1,:] = 0.5*(img[2:,:] - img[:-2,:])
        img_dxx = np.zeros(img.shape)
        img_dxx[:,1:-1] = 0.5*(img_dx[:,2:] - img_dx[:,:-2])
        img_dyy = np.zeros(img.shape)
        img_dyy[1:-1,:] = 0.5*(img_dy[2:,:] - img_dy[:-2,:])
        img_dxy = np.zeros(img.shape)
        img_dxy[:,1:-1] = 0.5*(img_dy[:,2:] - img_dy[:,:-2])

        grad = np.sqrt(img_dy**2 + img_dx**2)
        D = img_dxx*img_dyy - img_dxy**2
        return grad, D
        
    ellip, theta = toEllipse(xximg, yyimg, xyimg)
    cent = np.sqrt(ximg**2 + yimg**2)
    grad, lap = gradient(img)
    grad /= norm
    lap /= norm
    
    fellip, feta = toEllipse(xxf, yyf, xyf)
    fcent = np.sqrt(xf**2 + yf**2)
    xgrad, xlap = gradient(fakex)
    xlap = xlap[dy//2+1,dx//2+1]/norm_sum
    
    print fellip, feta
    #fellip = 0.35
    print ellip.shape, ellip.mean(), ellip.std()

    wy,wx = np.where(
        (ellip < fellip+feps) & (ellip > fellip-feps)
        & (img > lumcut*sigma)
        & (np.abs(cent) < centcut)
        & (grad < gradcut)
        & (lap < lapcut)
    )
    print "cent", cent.mean(), cent.std(), cent[wy,wx].mean(), cent[wy,wx].std()
    print "grad", grad.mean(), grad.std(), grad[wy,wx].mean(), grad[wy,wx].std()
    print "lap",  lap.mean(),  lap.std(),  lap[wy,wx].mean(),  lap[wy,wx].std(), xlap
    
    px, py = 4,3
    fig = figure.Figure(figsize=(8,8))
    canvas = FigCanvas(fig)
    for i, im in enumerate((soft, xxkern, yykern, xykern, ellip, theta,fake)):
        ax = fig.add_subplot(px,py,i+1)
        ax.imshow(im)
    i+=1
    ax = fig.add_subplot(px,py, i+1)

    stride = 8
    
    ax.plot(ellip.ravel()[::stride], lap.ravel()[::stride], '.k', markersize=0.1, alpha=0.2)
    ax.set_title("lap")
    ax.set_xlim([0.4, 0.8]) #fellip-0.1, fellip+0.1])
    ax.set_ylim([-1, 1])
    i+=1
    ax = fig.add_subplot(px,py, i+1)
    ax.plot(ellip.ravel()[::stride], cent.ravel()[::stride], '.k', markersize=0.1, alpha=0.2)
    ax.set_xlim([0.4, 0.8])
    ax.set_ylim([0.0, 5.0])
    ax.set_title("centroid")
    fig.savefig("moment.png")


    fig = figure.Figure()
    canvas = FigCanvas(fig)
    ax = fig.add_subplot(1,1,1)
    ax.plot(ellip.ravel()[::stride],theta.ravel()[::stride], 'k.', markersize=0.1, alpha=0.2)
    fig.savefig("evt.png")
    #ax.hist(ellip.ravel(), bins=50)
    
    fig = figure.Figure(figsize=(6,8))
    canvas = FigCanvas(fig)
    for i, im in enumerate((ellip, theta)):
        ax = fig.add_subplot(3,1,i+1)
        qq = ax.imshow(im)
        cax = fig.colorbar(qq)
    fig.savefig("ellip.png")
    
    #####################################
    # Hesse form
    #####################################
    
    # hesse form
    print "hesse"
    theta_tmp0 = theta[wy,wx].ravel() + np.pi/2.0
    theta_tmp = theta_tmp0.copy()
    xx = xx0[wy,wx].ravel()
    yy = yy0[wy,wx].ravel()
    
    r   = xx*np.cos(theta_tmp) + yy*np.sin(theta_tmp)
    neg = np.where(r < 0.0)[0]
    theta_tmp[neg] += np.pi
    cycle = np.where(theta_tmp > 2.0*np.pi)[0]
    theta_tmp[cycle] -= 2.0*np.pi
    r   = xx*np.cos(theta_tmp) + yy*np.sin(theta_tmp)

    #####################################
    # locate loci in hesse space
    #####################################

    print "Warning: length of inputs is ", len(theta_tmp)
    r_max = max(xx.max(), yy.max())
    r_new, t_new, _r, _xx, _yy = hesse.hesse_iter(theta_tmp, xx, yy, niter=2)
    bin2d, r_edge, t_edge, rs, ts, idx = hesse.hesse_bin(r_new, t_new, bins=bins, r_max=r_max, ncut=ncut, navg=navg)
    
    
    numLocus = len(ts)
    xfin = []
    yfin = []
    for i in range(numLocus):
        print "Locus ", i, ts[i], rs[i], len(idx[i])
        xfin.append(xx[idx[i]])
        yfin.append(yy[idx[i]])
    offset = 60.0

    prod_final = cent #lap #grad #cent
    #################################################################################
    #################################################################################
    print "scatter"
    fig = figure.Figure() #figsize=(8,10))
    fx, fy = 5, 2
    canvas = FigCanvas(fig)
    ax = fig.add_subplot(fy,fx,1)
    ax.set_title("1")
    print theta_tmp.shape, r.shape, prod_final[wy,wx].shape
    ax.scatter(theta_tmp, r, s=1.0, c=np.tile(prod_final[wy,wx],tiles), edgecolor='none', alpha=0.1)
    ax.set_xlim([0, 1.0*np.pi])
    ax.set_ylim([0, 4096])
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("$r$")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_size('xx-small')
    
    #################################################################################
    ax = fig.add_subplot(fy,fx,2)
    colors = ['g', 'r', 'b', 'k', 'c', 'm']
    ax.set_title("2")
    #ax.scatter(xx0[wy,wx], yy0[wy,wx], c=np.tile(prod_final[wy,wx], tiles), edgecolor='none', alpha=0.1, s=0.2)
    ax.scatter(xx0[wy,wx], yy0[wy,wx], c='k', edgecolor='none', alpha=0.1, s=0.2)
    for i in range(numLocus):
        ax.plot(xfin[i]+offset, yfin[i]+offset, '-'+colors[i%6])
        ax.plot(xfin[i]-offset, yfin[i]-offset, '-'+colors[i%6])
    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([0, img.shape[0]])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_size('xx-small')

    ax = fig.add_subplot(fy,fx,3)
    ax.set_title("3")
    ax.imshow(img, cmap='gray', vmin=-50, vmax=200, interpolation=None, origin='lower')
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_size('xx-small')
    
    #################################################################################
    #ax = fig.add_subplot(fy,fx,3)
    #ax.hist(theta)
    cx, cy = [0.29] + t_in, [1580.0] + r_in
    hx, hy = 0.15, 100.0

    for i in range(len(cy)):
        xmin = cx[i] - hx
        xmax = cx[i] + hx
        ymin = cy[i] - hy
        ymax = cy[i] + hy
        ax = fig.add_subplot(fy,fx,4+i)
        ax.set_title("%d"%(4+i))
        ax.hlines(r_edge, xmin, xmax)
        ax.vlines(t_edge, ymin, ymax)
        for j in range(numLocus):
            rect = Rectangle( (ts[j]-0.1, rs[j]-10), 0.2, 20, facecolor="none", edgecolor='red')
            ax.add_patch(rect)
        ax.imshow(bin2d, cmap='gray_r', origin='bottom', extent=(0.0, np.pi, 0.0, r_max), aspect='auto', interpolation='none')
        ax.scatter(theta_tmp, r, s=4.0, c=np.tile(prod_final[wy,wx], tiles), edgecolor='none')
        ax.scatter(ts, rs, s=100.0, marker='o', c='r', facecolor='none')
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("$r$")
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_size('xx-small')
            
            
    fig.savefig("foo.png")


    img0[:] = img
    # put the settings in a dict object and call ds9.mtv()
    settings = {'scale':scale, 'zoom': zoom, 'mask' : 'transparency %d' %(trans)}
    #ds9.mtv(exposure, frame=frame, title=title, settings=settings)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of data repository")
    parser.add_argument("visit", type=int, help="Visit to show")
    parser.add_argument("ccd", type=int, help="CCD to show")
    parser.add_argument("-f", "--frame", type=int, default=1, help="Frame")
    parser.add_argument("-s", "--scale", default="zscale", help="Gray-scale")
    parser.add_argument("-t", "--title", default="", help="Figure title")
    parser.add_argument("-T", "--trans", default=100, help="Transparency")
    parser.add_argument("-z", "--zoom",  default="to fit", help="Zoom")
    args = parser.parse_args()

    main(args.root, args.visit, args.ccd,
         frame=args.frame, title=args.title, scale=args.scale, zoom=args.zoom, trans=args.trans
     )
