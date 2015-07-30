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

def main(infile, selectfile):

    lines = []
    with open(selectfile, 'r') as fp:
        lines = fp.readlines()

    lookup = {"sat" : "satellite", "air" : "aircraft", "mou":"moustache",
              "sca": "scattered", "dif": "diffraction", "swa" : "swallow", "unk": "unknown"}

    typs = {}
    print "loading selections"
    for line in lines:
        fields = line.split("QQ")
        png = fields[1].strip()
        typ = fields[3].strip()

        m = re.match("det-(\d+)-(\d+).png", png)
        if m:
            (v, c) = [int(x) for x in m.groups()]
            typs[(v,c)] = typ
        else:
            continue
        
    
    print "loading"
    lines = []
    with open(infile, 'r') as fp:
        lines = fp.readlines()

    print "parsing"
    detections = collections.defaultdict(list)
    seen = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        
        m = re.match("\((\d+), (\d+)\) SatelliteTrail\(r=(\d+.\d),theta=(\d.\d+),width=(\d+.\d+),.*", line)
        if m:
            v, c, r, t, w = m.groups()
            v = int(v)
            c = int(c)
            r = float(r)
            t = float(t)
            w = float(w)
            k = lookup[typs[(v,c)]]
            s = "Candidate(\"%s\", %d, %d, satTrail.SatelliteTrail(r=%.1f, theta=%.3f, width=%.1f)),"%(k,v,c,r,t,w)
            detections[(int(v),int(c))].append(s)

    for k,arr in sorted(detections.items()):
        for a in arr:
            print a
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="file with the list of detections in it")
    parser.add_argument("selectfile", help="file with the list of selections in it")
    args = parser.parse_args()
    main(args.infile, args.selectfile)
            
    
