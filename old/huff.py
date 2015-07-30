#!/usr/bin/env python

import argparse
import itertools
import numpy as np
from scipy import ndimage as ndimg

#import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

import lsst.daf.persistence  as dafPersist
import lsst.afw.image        as afwImage
import lsst.afw.display.ds9  as ds9

import lssttools.filter as filt
import addTrace

# try shorter baselines
# plot distances

def main(rootDir, visit, ccd, frame=1, title="", scale="zscale", zoom="to fit", trans=60):


    nrep = 3   # number of random points to use
    nfrac = 3  # intevening points
    nsig = 5.0 # clip limit for measuring image noise

    
    # make a butler and specify your dataId
    butler = dafPersist.Butler(rootDir)
    dataId = {'visit': visit, 'ccd':ccd}

    # get the exposure from the butler
    exposure = butler.get('calexp', dataId)

    img0 = exposure.getMaskedImage().getImage().getArray()#[:1024,:]


    #####################################
    # Add fakes
    ######################################
    
    r_in = [500.0, 1000.0, 500.0, 2000.0]
    t_in = [0.28, 0.28, 0.28+np.pi/2.0, 0.28]
    flux = [70.0, 80.0, 90.0, 100.0]
    imgx = img0
    for i in range(len(r_in)):
        imgx = addTrace.addTrace(imgx, r_in[i], t_in[i], flux[i])


    img = filt.fftSmooth2d(imgx, 4.0, 4.0, 0.0)

    #####################################
    # measure image noise
    #####################################
    
    sigma = img.std()
    imclip = np.clip(img, -nsig*sigma, nsig*sigma)
    sigma = imclip.std()
        

    ######################################
    # Pick random coordinates
    ######################################
    img_orig = img.copy()

    xx0, yy0 = np.meshgrid(np.arange(img.shape[1], dtype=int), np.arange(img.shape[0], dtype=int))

    theta = np.array([])
    r = np.array([])
    xhit = np.array([])
    yhit = np.array([])
    dhit = np.array([])
    wxs = np.array([], dtype=int)
    wys = np.array([], dtype=int)
    wx, wy = None, None
    gx, gy = None, None
    prod_final = None
    p = None
    for i_rep in range(nrep):

        if True: #i_rep == 0:
            xx, yy = xx0, yy0
        else:
            xx = np.random.choice(gx, xx.shape)
            yy = np.random.choice(gy, yy.shape)
            xx += np.random.uniform(10, size=img.shape).astype(int) - 5
            yy += np.random.uniform(10, size=img.shape).astype(int) - 5
            xx = np.clip(xx, 0, img.shape[1])
            yy = np.clip(yy, 0, img.shape[0])
            
        yran = np.random.uniform(img.shape[0], size=img.shape).astype(int)
        xran = np.random.uniform(img.shape[1], size=img.shape).astype(int)
            
        stack = [img]
        for i in range(1, nfrac):
            frac = i/(1.0*nfrac)
            print frac
            ymid = ((1.0-frac)*yy + frac*yran).astype(int)
            xmid = ((1.0-frac)*xx + frac*xran).astype(int)
            immid = img[ymid,xmid]
            #immid = np.clip(img[ymid,xmid], 0.0, None)
            stack.append(immid)
        imran = img[yran,xran]
        #imran = np.clip(img[yran,xran], 0.0, None)
        stack.append(imran)
        stack = np.array(stack)

        d2 = (xx - xran)**2 + (yy - yran)**2

        print "clip"

        clip = 0.0 #-0.5*sigma
        prod = np.ones(stack[0].shape)
        for i in range(len(stack)):
            #prod *= stack[i,:,:]
            prod *= np.clip(stack[i,:,:], clip, None)
        prod = np.clip(prod, 0.0, None)

        if prod_final is None:
            prod_final = prod
        else:
            prod_final *= prod
        
        print prod.min(), prod.max()
        mean = stack.mean(axis=0)
        print mean.shape
        std = stack.std(axis=0)
        #mean_std = std.mean()
        #z = np.abs( (stack-mean) / std).max(axis=0)

        #print z.shape
        wy,wx = np.where( (prod > (1.0*sigma)**(nfrac+1)) &
                          (mean > 0.5*sigma) &
                          (d2 > 400**2) &
                          (std < 2.0*sigma) )
        wxs = np.append(wxs, wx)
        wys = np.append(wys, wy)

        gx = np.append(xx[wy,wx], xran[wy,wx])
        gy = np.append(yy[wy,wx], yran[wy,wx])

        p = np.append(prod[wy, wx], prod[wy,wx])
        p /= p.sum()
        
        # hesse form
        print "hesse"
        theta_tmp = np.arctan2( -(xran[wy,wx] - xx[wy,wx]), (yran[wy,wx] - yy[wy,wx]))
        r_tmp = xx[wy,wx]*np.cos(theta_tmp) + yy[wy,wx]*np.sin(theta_tmp)
        neg = np.where(r_tmp < 0.0)[0]
        theta_tmp[neg] += np.pi
        theta = np.append(theta, theta_tmp)
        r     = np.append(r, xx[wy,wx]*np.cos(theta_tmp) + yy[wy,wx]*np.sin(theta_tmp))

        xhit = np.append(xhit, xx[wy,wx])
        yhit = np.append(yhit, yy[wy,wx])
        dhit = np.append(dhit, np.sqrt(d2[wy,wx]))

        print sigma, len(wx), len(wy), r.shape

    wx = wxs
    wy = wys

    bins = 200
    bin2d, theta_edge, r_edge = np.histogram2d(theta, r, bins=bins, range=((0.0, np.pi),(0.0, max(img.shape))))
    wypos, wxpos = np.where(bin2d > 0)
    avgPerBin = np.mean(bin2d[wypos,wxpos])
    thresh = max(10.0, 5*avgPerBin)
    print "Threshold: ", len(wypos), len(theta), avgPerBin, thresh
    locus, numLocus = ndimg.label(bin2d > thresh, structure=np.ones((3,3)))

    print "Loci: ", numLocus
        
    r_best     = np.array([])
    theta_best = np.array([])
    xfin = []
    yfin = []
    dfin = []

    for i in range(numLocus):

        loc_t,loc_r = np.where(locus == i + 1)
        tlo, thi = theta_edge[loc_t.min()-1], theta_edge[loc_t.max()+1]
        rlo, rhi = r_edge[loc_r.min()-1], r_edge[loc_r.max()+1]

        wtmp = np.where((theta > tlo) & (theta < thi) & (r > rlo) & (r < rhi))[0]
        print loc_t, loc_r, tlo, thi, rlo, rhi, len(wtmp)
        xfin.append(xhit[wtmp])
        yfin.append(yhit[wtmp])
        dfin.append(dhit[wtmp])
        r_best = np.append(r_best, r[wtmp].mean())
        theta_best = np.append(theta_best, theta[wtmp].mean())

    dx = np.cos(theta_best)
    dy = np.sin(theta_best)
    offset = 100.0

    #################################################################################
    #################################################################################
    print "scatter"
    fig = figure.Figure()
    canvas = FigCanvas(fig)
    ax = fig.add_subplot(241)
    ax.set_title("1")
    print theta.shape, r.shape, prod_final[wy,wx].shape
    ax.scatter(theta, r, s=1.0, c=prod_final[wy,wx], edgecolor='none', alpha=0.1)
    ax.set_xlim([0, 3.142])
    ax.set_ylim([0, 4096])
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("$r$")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_size('xx-small')
    
    #################################################################################
    ax = fig.add_subplot(242)
    ax.set_title("2")
    ax.scatter(xx[wy,wx], yy[wy,wx], c=prod_final[wy,wx], edgecolor='none', alpha=0.1)
    for i in range(numLocus):
        ax.plot(xfin[i]+offset*dx[i], yfin[i]+offset*dy[i], '-r')
        ax.plot(xfin[i]-offset*dx[i], yfin[i]-offset*dy[i], '-r')
    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([0, img.shape[0]])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_size('xx-small')

    ax = fig.add_subplot(243)
    ax.set_title("3")
    ax.imshow(img, cmap='gray', vmin=-50, vmax=200, interpolation=None, origin='lower')
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_size('xx-small')
    
    #################################################################################
    #ax = fig.add_subplot(243)
    #ax.hist(theta)
    cx, cy = [0.29] + t_in, [1580.0] + r_in
    hx, hy = 0.05, 40.0

    for i in range(len(cy)):
        xmin = cx[i] - hx
        xmax = cx[i] + hx
        ymin = cy[i] - hy
        ymax = cy[i] + hy
        ax = fig.add_subplot(244+i)
        ax.set_title("%d"%(4+i))
        ax.scatter(theta, r, s=4.0, c=prod_final[wy,wx], edgecolor='none')
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
    ds9.mtv(exposure, frame=frame, title=title, settings=settings)

    
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
