#!/usr/bin/env python

import sys, os, re
import argparse
import datetime
import collections
import numpy as np
import matplotlib.figure as figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas

import lsst.daf.persistence as dafPersist

import satelliteTrail as satTrail

def main(root, infile):

    butler = dafPersist.Butler(root)

    print "loading"
    lines = []
    with open(infile, 'r') as fp:
        lines = fp.readlines()

    print "parsing"
    detections = collections.defaultdict(list)
    for line in lines:
        m = re.match("\((\d+), (\d+)\) SatelliteTrail\(r=(\d+.\d),theta=(\d.\d+),width=(\d+.\d+),.*", line)
        if m:
            v, c, r, t, w = m.groups()
            detections[(int(v),int(c))].append( satTrail.SatelliteTrail(r=float(r),theta=float(t),width=float(w)) )

    print "plotting"
    for (v,c), trails in detections.items():
        print v, c
        
        dataId = {'visit': int(v), 'ccd': int(c)}
        cexp = butler.get('calexp', dataId)
        img = cexp.getMaskedImage().getImage().getArray()

        ny, nx = img.shape
        
        fig = figure.Figure()
        can = FigCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(np.arcsinh(img), origin='lower', cmap='gray')

        for trail in trails:
            x1, y1 = trail.trace(nx, ny, offset=40)
            x2, y2 = trail.trace(nx, ny, offset=-40)
            ax.plot(x1, y1, 'r-')
            ax.plot(x2, y2, 'r-')

        ax.set_xlim([0, nx])
        ax.set_ylim([0, ny])
        t0 = trails[0]
        ax.set_title("(%s) %s %s" % (str(datetime.datetime.now()), str((v,c)), str((t0.r,t0.theta,t0.width))))
        fig.savefig("det-%05d-%03d.png" % (v, c))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="butler root dir")
    parser.add_argument("infile", help="file with the list of detections in it")
    args = parser.parse_args()
    main(args.root, args.infile)
            
    
