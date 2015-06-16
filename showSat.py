#!/usr/bin/env python

import sys, os, re
import copy
import argparse
import collections
import datetime
import numpy as np
import matplotlib.figure as figure
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
import lssttools.utils as lsstutil

def main(rerun, visits):

    butler = lsstutil.getButler(rerun, root=None)
    
    bin = 32
    
    # load the known sat file
    filename = "testvisits.dat"
    with open(filename, 'r') as fp:
        lines = fp.readlines()

    all_types = "bfpvaF"
    ccds = {}
    tps, fps, tns, fns = {}, {}, {}, {}
    for c in all_types:
        ccds[c] = collections.defaultdict(list)
        tps[c], fps[c], tns[c], fns[c] = 0, 0, 0, 0
        
    for line in lines:
        line = re.sub('#.*', '', line).strip()
        if len(line) == 0:
            continue
        
        fields = line.split()

        v, f, n = fields[0:3]
        if int(n) > 0:
            types = fields[3].split(',')
            for i, t in enumerate(types):
                ccds[t][int(v)] += [int(re.sub("[cs]", "", c)) for c in fields[4+i].split(',')]

    for visit in visits:

        print "Running", visit
        
        logs = [(c, "data/%04d/log%05d-%03d.txt" % (visit, visit ,c)) for c in range(104)]
        all_positive = set()
        for c, log  in logs:
            if os.path.exists(log):
                with open(log, 'r') as lp:
                    for line in lp.readlines():
                        m = re.search("Detected ([123456789]) satellite", line)
                        if m:
                            all_positive.add(c)
        all_negative = set(range(104)) - all_positive

        fig = figure.Figure()
        canvas = FigCanvas(fig)
        fig.suptitle( str(visit) + "    "+ datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S") )
        cmap = cm.gist_rainbow
        cmap.set_under('w', 0.0)
        
        for i_t, track_type in enumerate(all_types):
        
            present  = set(ccds[track_type][visit])
            absent   = set(range(104)) - present

            # only consider positives if they're not expected in one of the other types
            # for detection, we don't know what 'type' we got.
            positive = copy.copy(all_positive)
            for i_tt, track_tmp in enumerate(all_types):
                if i_t == i_tt:
                    continue
                # in this type, but not in our type
                s = set(ccds[track_tmp][visit]) - set(ccds[track_type][visit])                
                positive -= s
            negative = set(range(104)) - positive
            
            # make plot
            fimg = lsstutil.FpaImage(butler.get('camera'), scale=bin)

            tp = present & positive, 1.8
            tn = absent  & negative, 2.4
            fn = present & negative, 4.0
            fp = absent  & positive, 0.1

            tps[track_type] += len(tp[0])
            tns[track_type] += len(tn[0])
            fps[track_type] += len(fp[0])
            fns[track_type] += len(fn[0])


            fimg.image -= 0.1
            for s,v in (tp, tn, fn, fp):
                for c in s:
                    img = fimg.getPixels(c)
                    img += v

            ax = fig.add_subplot(2,3,i_t + 1)
            ax.set_title("Type %s" % (track_type))
            ax.imshow(fimg.image[::-1], cmap=cmap, vmin=0.0, vmax=4.0)

        png = "fpa-%05d.png" % (visit)
        fig.savefig(png)

    for track_type in all_types:
        
        n_positive = tps[track_type] + fps[track_type] + 0.001
        n_trail    = tps[track_type] + fns[track_type] + 0.001
        recall    = 1.0*tps[track_type]/ n_trail
        precision = 1.0*tps[track_type] / n_positive

        f1 = 2.0 * (recall*precision)/(recall + precision + 0.001)

        print "Final (type=%s) Precision/Recall/f1:" % (track_type), precision, recall, f1


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rerun", help="rerun")
    parser.add_argument("visit", help="Visit number")
    args = parser.parse_args()
    main(args.rerun, [int(x) for x in lsstutil.idSplit(args.visit)])
