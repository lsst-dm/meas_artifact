#!/usr/bin/env python

import argparse
import lsst.daf.persistence as dafPersist
import hsc.pipe.base.butler as hscButler
import satelliteTrail as satTrail
import SatelliteTask as satTask



colors = {
    "red"    :"31",
    "green"  :"32",
    "yellow" :"33",
    "blue"   :"34",
    "magenta":"35",
    "cyan"   :"36",
    "grey"   :"37",
    }


def color(text, color, bold=False):
    base = "\033["
    code = colors[color]
    if bold:
        code += ";1"
    prefix = base + code + "m"
    suffix = base + "0m"
    return prefix + text + suffix


class Candidate(object):
    def __init__(self, visit, ccd, trail):
        self.visit = visit
        self.ccd = ccd
        self.trail = trail

        
knownCandidates = [
    
    # satellites
    Candidate(1236, 65, satTrail.SatelliteTrail(r=1580.0, theta=0.286, width= 7.94)),
    Candidate( 242, 95, satTrail.SatelliteTrail(r=1497.8, theta=1.245, width=21.12)),
    Candidate( 270, 78, satTrail.SatelliteTrail(r=1195.6, theta=5.871, width=14.58)),
    Candidate(1184, 78, satTrail.SatelliteTrail(r=2218.0, theta=0.841, width=19.96)),
    Candidate(1168, 47, satTrail.SatelliteTrail(r=2492.8, theta=1.430, width=12.91)),
    Candidate(1166, 96, satTrail.SatelliteTrail(r=1177.4, theta=6.162, width=11.57)),

    # aircraft
    Candidate(1166, 65, satTrail.SatelliteTrail(r= 791.9, theta=6.058, width=22.55)),
    Candidate(1166, 70, satTrail.SatelliteTrail(r= 244.4, theta=2.905, width=25.09)),
    Candidate(1188, 18, satTrail.SatelliteTrail(r=1140.0, theta=6.183, width=18.73)),
    Candidate(1240, 51, satTrail.SatelliteTrail(r= 885.4, theta=2.379, width=19.34)),
    Candidate(1248, 43, satTrail.SatelliteTrail(r= 606.7, theta=6.013, width=15.81)),
]


def main(root):

    allMessages = ""
    for cand in knownCandidates:

        butler       = dafPersist.Butler(root)
        dataId       = {'visit': cand.visit, 'ccd': cand.ccd}
        dataRef      = hscButler.getDataRef(butler, dataId)
        
        task         = satTask.SatelliteTask()
        foundTrails  = task.run(dataRef)

        rMax, thetaMax = 20.0, 0.15
        t = cand.trail
        result = ""
        claimed = [False]*len(foundTrails)
        for iTrail,fTrail in enumerate(foundTrails):
            if fTrail.isNear(t, rMax, thetaMax):
                result += "\n  %s: %s" % (color("FOUND", "green"), fTrail)
                claimed[iTrail] = True
        if not result:
            result += "\n  %s!!" % (color("MISSING", "red"))
            
        nUnclaimed = len(claimed) - sum(claimed)
        if nUnclaimed > 0:
            result += "\n  %s: %d Unclaimed trails (total=%d).\n" % \
                      (color("WARNING", "yellow"), nUnclaimed, len(foundTrails))
            for iClaim,claim in enumerate(claimed):
                if not claim:
                    result  += "  --> Unclaimed: %s\n" % (color(str(foundTrails[iClaim]), 'yellow'))

        did = "(%d, %d)" % (cand.visit, cand.ccd)
        msg = "%s [r=%.1f theta=%.3f wid=%.1f]." % (color(did, "cyan"), t.r, t.theta, t.width)
        msg += result
        allMessages += msg+"\n"
        print "\n"+msg+"\n"

    print color("=== Summary ===", "magenta")
    print allMessages


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    main(args.root)
