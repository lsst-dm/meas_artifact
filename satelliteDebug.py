
import datetime
import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
from matplotlib.patches import Rectangle


def font(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_size("xx-small")

colors = 'm', 'c', 'g', 'r'
dr = 100
dt = 0.2
py, px = 2, 4


def coordPlot(exposure, finder, pngfile):

    mm = finder._mm
    cmm = finder._mmCals
    img = exposure.getMaskedImage().getImage().getArray()
    bins = 1.0*img.shape[1]/mm.img.shape[1]
    xx, yy = np.meshgrid(np.arange(mm.img.shape[1], dtype=int), np.arange(mm.img.shape[0], dtype=int))
    x = (bins*xx[finder._isCandidate]).astype(int)
    y = (bins*yy[finder._isCandidate]).astype(int)

    cimg = np.zeros(img.shape)
    cimg[y,x] = 1.0


    def small(ax):
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_size("small")
            
    fig = figure.Figure(figsize=(8.0,4.0))
    fig.subplots_adjust(bottom=0.15)
    can = FigCanvas(fig)
    
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(np.arcsinh(img), origin='lower', cmap='gray')
    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([0, img.shape[0]])
    small(ax)
    
    ax = fig.add_subplot(1, 3, 3)
    stride = 1
    #ax.plot(mm.theta[::stride], mm.ellip[::stride], '.k', ms=0.2, alpha=0.2)
    ax.scatter(mm.theta[::stride], mm.ellip[::stride],
               c=np.clip(mm.center[::stride], 0.0, 4.0*finder.centerLimit),
               s=0.2, alpha=0.2, edgecolor='none')
    if len(cmm) == 2:
        i = 0
    else:
        i = 2
    ax.hlines([cmm[i].ellip], -np.pi/2.0, np.pi/2.0, color='k', linestyle='-')
    ax.set_xlim([-np.pi/2.0, np.pi/2.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("$e=1-B/A$")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    small(ax)
    
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(cimg, origin='lower', cmap='gray_r')
    ax.plot(x, y, 'k.', ms=1.0)
    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([0, img.shape[0]])
    small(ax)
    
    fig.savefig(pngfile)
    fig.savefig(re.sub("png", "eps", pngfile))
    

def debugPlot(finder, pngfile):

    mm     = finder._mm
    cmm    = finder._mmCals
    trails = finder._trails

    img = mm.img

    ###################################
    # a debug figure
    ###################################
    debug = True
    if debug:


        fig = figure.Figure()
        can = FigCanvas(fig)
        fig.suptitle(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # pixel plot
        ax = fig.add_subplot(py, px, 1)
        pixelPlot(finder, ax, mm, cmm, trails)
        
        # hough  r vs theta
        ax = fig.add_subplot(py, px, 2)
        houghPlot(finder, ax, mm, cmm, trails)
        
        # e vs theta
        ax = fig.add_subplot(py, px, 3)
        evthetaPlot(finder, ax, mm, cmm, trails)

        # centroid vs flux
        ax = fig.add_subplot(py, px, 4)
        cvfluxPlot(finder, ax, mm, cmm, trails)

        # b versus skew
        ax = fig.add_subplot(py, px, 5)
        bvskewPlot(finder, ax, mm, cmm, trails)

        # trail plots
        for i,trail in enumerate(trails[0:px*py-5]):
            ax = fig.add_subplot(py,px,6+1*i)
            trailPlot(finder, ax, mm, cmm, trail, i)
        
        fig.savefig(pngfile)


 
        
def pixelPlot(finder, ax, mm, cmm, trails):
    xx, yy = np.meshgrid(np.arange(mm.img.shape[1], dtype=int), np.arange(mm.img.shape[0], dtype=int))

    ax.imshow(np.arcsinh(mm.img), cmap="gray", origin='lower')
    ax.scatter(xx[finder._isCandidate], yy[finder._isCandidate], c=mm.theta[finder._isCandidate], s=3.0, edgecolor='none', vmin=-np.pi, vmax=np.pi, cmap='rainbow')
    ny, nx = mm.img.shape
    for i,trail in enumerate(trails):
        x, y = trail.trace(nx, ny, offset=30, bins=finder.bins)
        ax.plot(x, y, colors[i%4]+'-')
        x, y = trail.trace(nx, ny, offset=-30, bins=finder.bins)
        ax.plot(x, y, colors[i%4]+'-')
    ax.set_xlim([0, nx])
    ax.set_ylim([0, ny])
    font(ax)

def houghPlot(finder, ax, mm, cmm, trails):
    
    ny, nx = mm.img.shape
    ax.plot(finder._solutions.theta, finder.bins*finder._solutions.r, 'k.', ms=1.0, alpha=0.1)
    for i,trail in enumerate(trails):
        ax.plot(trail.theta, trail.r, 'o', mfc='none', mec=colors[i%4], ms=10)
        ax.add_patch(Rectangle( (trail.theta - dt, trail.r - finder.bins*dr),
                                2*dt, 2*finder.bins*dr, facecolor='none', edgecolor=colors[i%4]))
    ax.set_xlabel("Theta", size='small')
    ax.set_ylabel("r", size='small')
    ax.set_xlim([0.0, 2.0*np.pi])
    ax.set_ylim([0.0, finder.bins*ny])
    ax.text(0.95, 0.95, "N=%d" % (len(finder._solutions.theta)), size='xx-small',
            horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    font(ax)

def evthetaPlot(finder, ax, mm, cmm, trails):
    
    ny, nx = mm.img.shape
    stride = int(1.0*mm.theta.size/20000)
    if stride < 1:
        stride = 1
    #ax.plot(finder.theta[::stride], mm.ellip[::stride], '.k', ms=0.2, alpha=0.2)
    ax.scatter(mm.theta[::stride], mm.ellip[::stride],
               c=np.clip(mm.center[::stride], 0.0, 2.0*finder.centerLimit),
               s=0.4, alpha=0.4, edgecolor='none')
    ax.scatter(mm.theta[finder._isCandidate], mm.ellip[finder._isCandidate],
               c=np.clip(mm.center[finder._isCandidate], 0.0, 2.0*finder.centerLimit),
               s=0.8, alpha=0.8, edgecolor='none')
    for i in range(len(cmm)):
        ax.hlines([cmm[i].ellip], -np.pi/2.0, np.pi/2.0, color='m', linestyle='-')
        fhlines = [
            cmm[i].ellip - finder.eRange,
            cmm[i].ellip + finder.eRange,
        ]
        bhlines = [
            cmm[i].ellip - finder._brightFactor*finder.eRange,
            cmm[i].ellip + finder._brightFactor*finder.eRange,
        ]
        ax.hlines(fhlines, -np.pi/2.0, np.pi/2.0, color='m', linestyle='--')
        ax.hlines(bhlines, -np.pi/2.0, np.pi/2.0, color='c', linestyle='--')
    for i,trail in enumerate(trails):
        x, y = trail.trace(nx, ny, offset=0, bins=finder.bins)
        x = x.astype(int)
        y = y.astype(int)
        ax.plot(mm.theta[y,x], mm.ellip[y,x], colors[i%4]+'.', ms=1.0)
    ax.set_xlabel("Theta", size='small')
    ax.set_ylabel("e", size='small')
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([0.0, 1.0])
    #ax.set_ylim([finder.ellipCal-3.0*finder.eRange, finder.ellipCal+3.0*finder.eRange])
    font(ax)

def cvfluxPlot(finder, ax, mm, cmm, trails):
    
    ny, nx = mm.img.shape
    stride = int(1.0*mm.theta.size/20000)
    ax.scatter(mm.center[::stride], mm.sumI[::stride]/mm.std,
               c=np.clip(mm.center_perp[::stride], 0.0, 2.0*finder.centerLimit), marker='.', s=1.0,
               alpha=0.5, edgecolor='none')
    ax.scatter(mm.center[finder._isCandidate], mm.sumI[finder._isCandidate]/mm.std,
               c=np.clip(mm.center_perp[finder._isCandidate], 0.0, 2.0*finder.centerLimit),
               s=2.0, alpha=1.0, edgecolor='none')
    for i,trail in enumerate(trails):
        x, y = trail.trace(nx, ny, offset=0, bins=finder.bins)
        x = x.astype(int)
        y = y.astype(int)
        ax.plot(mm.center[y,x], mm.sumI[y,x]/mm.std, colors[i%4]+'.', ms=1.0)

    ax.vlines([finder.centerLimit, finder.centerLimit/finder._brightFactor],
              finder.luminosityLimit/10.0, 10000.0*finder.luminosityLimit, color='k', linestyle='--')
    ax.hlines([finder.luminosityLimit], 0.01, 10, color='k', linestyle='--')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Center", size='small')
    ax.set_ylabel("Flux", size='small')
    ax.set_xlim([0.01, 10])
    ax.set_ylim([finder.luminosityLimit/10.0, 10000.0*finder.luminosityLimit])
    font(ax)


def bvskewPlot(finder, ax, mm, cmm, trails):
    
    ny, nx = mm.img.shape
    for i in range(len(cmm)):
        ax.scatter(mm.skew, mm.b-cmm[i].b,
                   c=np.clip(mm.center, 0.0, 2.0*finder.centerLimit), marker='.', s=1.0,
                   alpha=0.5, edgecolor='none')
        ax.scatter(mm.skew[finder._isCandidate], mm.b[finder._isCandidate]-cmm[i].b,
                   c=np.clip(mm.center[finder._isCandidate], 0.0, 2.0*finder.centerLimit), marker='.', s=2.0,
                   alpha=1.0, edgecolor='none')
    ax.vlines([finder.skewLimit, finder.skewLimit/finder._brightFactor], 0, 3.0, linestyle='--', color='k')
    fhlines = [ -finder.bLimit, finder.bLimit ]
    bhlines = [ -finder._brightFactor*finder.bLimit, finder._brightFactor*finder.bLimit ]
    ax.hlines(fhlines, 0, 3.0*finder.skewLimit, linestyle='--', color='m')
    ax.hlines(bhlines, 0, 3.0*finder.skewLimit, linestyle='--', color='c')
    for i,trail in enumerate(trails):
        x, y = trail.trace(nx, ny, offset=0, bins=finder.bins)
        x = x.astype(int)
        y = y.astype(int)
        marker = '.', '+', 'o'
        for j in range(len(cmm)):
            ax.plot(mm.skew[y,x], mm.b[y,x] - cmm[j].b, colors[i%4]+marker[j], ms=4.0)
    ax.set_xlabel("Skew", size='small')
    ax.set_ylabel("B", size='small')
    ax.set_xlim([0.0, 3.0*finder.skewLimit])
    ax.set_ylim([-2.0, 4.0])
    font(ax)


def trailPlot(finder, ax, mm, cmm, trail, i):
    
    ax.plot(finder._solutions.theta, finder.bins*finder._solutions.r, 'k.', ms=1.0, alpha=0.8)
    ax.plot(finder._solutions[i].thetaNew, finder.bins*finder._solutions[i].rNew, 'r.', ms=1.0, alpha=0.8)
    ax.plot(trail.theta, trail.r, 'o', mfc='none', ms=20, mec=colors[i%4])
    ax.set_xlabel("Theta", size='small')
    ax.set_ylabel("r", size='small')
    rmin, rmax = trail.r - dr, trail.r + dr
    tmin, tmax = trail.theta - dt, trail.theta + dt
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([rmin, rmax])
    w = (np.abs(finder._solutions.theta - trail.theta) < dt) & \
        (np.abs(finder.bins*finder._solutions.r - trail.r) < dr)
    wNew = (np.abs(finder._solutions[i].thetaNew - trail.theta) < dt) & \
           (np.abs(finder.bins*finder._solutions[i].rNew - trail.r) < dr)
    ax.text(0.95, 0.95, "N=%d" % (w.sum()), size='xx-small',
            horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.95, 0.90, "N=%d" % (wNew.sum()), size='xx-small',color = 'r',
            horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)
    font(ax)

