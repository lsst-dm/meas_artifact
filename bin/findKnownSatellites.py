#!/usr/bin/env python

import argparse
import collections
import cPickle as pickle
import numpy as np

import lsst.daf.persistence as dafPersist

import lsst.meas.artifact as measArt

import lsst.meas.artifact.mapreduce      as mapr
import lsst.meas.artifact.candidates     as candi
import lsst.meas.artifact.colors         as clr


candidateSets = {
    'all'   : candi.knownCandidates,
    'short' : candi.shortCandidates,
}


def hashDataId(dataId):
    return  (int(dataId['visit']), int(dataId['ccd']))
    
def process(dataRef):
    config          = measArt.HoughSatelliteConfig()
    #config.doBroad  = False
    task            = measArt.HoughSatelliteTask(config=config)
    exposure        = dataRef.get("calexp", immediate=True)
    trails, runtime = task.process(exposure)
    return (hashDataId(dataRef.dataId), trails, runtime)

    
Event = collections.namedtuple("Event", "input detected")
class EventList(list):

    @property
    def positiveDetections(self):
        return 1.0*sum(1 for input,detected in self if detected)
    @property
    def negativeDetections(self):
        return 1.0*sum(1 for input,detected in self if not detected)
    @property
    def positiveInputs(self):
        return 1.0*sum(1 for input,detected in self if input)
    @property
    def negativeInputs(self):
        return 1.0*sum(1 for input,detected in self if not input)
        
    @property
    def truePositives(self):
        return 1.0*sum(1 for input,detected in self if input and detected)
    @property
    def trueNegatives(self):
        return 1.0*sum(1 for input,detected in self if (not input) and (not detected))
    @property
    def falsePositives(self):
        return 1.0*sum(1 for input,detected in self if (not input) and detected)
    @property
    def falseNegatives(self):
        return 1.0*sum(1 for input,detected in self if input and (not detected))

    @property
    def recall(self):
        posIn = self.positiveInputs
        if posIn == 0:
            posIn = -1
        return self.truePositives/posIn
    @property
    def precision(self):
        posDet = self.positiveDetections
        if posDet == 0:
            posDet = -1
        return self.truePositives/posDet
    @property
    def f1(self):
        denom = self.recall + self.precision
        if len(self) == 0 or denom == 0.0:
            return 0.0
        return 2.0*(self.recall*self.precision)/denom
        



        
def main(root, threads, output, input=None, kind=None, visit=None, sensor=None, candidateSet=None):

    if not candidateSet:
        candidateSet = candi.knownCandidates
        
    butler       = dafPersist.Butler(root)
    rMax, thetaMax = 100.0, 0.15
        
    allMessages = ""

    ###################################################################
    # Create an eventList for all kinds of candiate Trail
    eventLists = {}
    nMasked = {}
    for k in candi.Candidate.validKinds:
        eventLists[k] = EventList()
        nMasked[k] = 0
    eventLists['all'] = EventList()
    nMasked['all'] = 0

    candidateLookup = collections.defaultdict(list)
    for candidate in candidateSet:
        did = {'visit': candidate.visit, 'ccd':candidate.ccd }
        candidateLookup[hashDataId(did)].append( candidate )

    
    ###################################################################
    # Add jobs to the map    
    if input is None:
        mp = mapr.MapFunc(threads, process)
        alreadyProcessing = set()
        for candidate in candidateSet:
            rightKind  = kind  is None or (candidate.kind in kind)
            rightVisit = visit is None or (candidate.visit in visit)
            rightSensor = sensor is None or (candidate.ccd in sensor)
            if rightKind and rightVisit and rightSensor and \
                    (candidate.visit,candidate.ccd) not in alreadyProcessing:
                dataId       = {'visit': candidate.visit, 'ccd': candidate.ccd}
                dataRefList = [ref for ref in butler.subset("calexp", **dataId)]
                assert len(dataRefList) == 1
                dataRef = dataRefList.pop()
                if not dataRef.datasetExists():
                    continue
                mp.add(dataRef)
                alreadyProcessing.add( (candidate.visit, candidate.ccd) )
        results = mp.run()

    else:
        with open(input, 'r') as fp:
            results = pickle.load(fp)

    if output is not None:
        with open(output, 'w') as fp:
            pickle.dump(results, fp)


    if kind is not None:
        resultsTmp = []
        for result in results:
            dataHash, foundTrails, runtime = result
            candidates = candidateLookup[dataHash]
            rightKind = False
            for candidate in candidates:
                if candidate.kind in kind:
                    rightKind = True
            if rightKind:
                resultsTmp.append(result)
        results = resultsTmp

    
    falsePos, falseNeg = [], []

    
    ####################################################################
    # Tally the results and see how we did
    runtimes = []
    widths = collections.defaultdict(list)
    
    for result in results:
        dataHash, foundTrails, runtime = result
        runtimes.append(runtime)

        # if there's no candidate for this data
        candidates = candidateLookup[dataHash]
        if len(candidates) == 0:

            resultMsg = ""
            # if we found something it's a false positive
            if len(foundTrails) > 0:
                for iTrail, fTrail in enumerate(foundTrails):                
                    resultMsg  += "\n  %s: %s (%s)" % (clr.color("FALSE-POS", "red"), fTrail, "no-candidate")
                    eventLists['empty'].append(Event(False, True))
                    eventLists['all'].append(Event(False, True))
                    falsePos.append((dataHash, fTrail))
                    
            # otherwise, it's a true negative
            else:
                resultMsg  += "\n  %s: %s (%s)" % (clr.color("TRUE-NEG", "green"), "No Trail", "no-candidate")
                eventLists['empty'].append(Event(False, False))
                eventLists['all'].append(Event(False, False))
            allMessages += resultMsg
                
        
        for candidate in candidateLookup[dataHash]:

            nTrail = len(foundTrails)
            
            eList = EventList()

            t = candidate.trail
            resultMsg = ""
            #########################################
            # True positives - does result match candidate
            #########################################
            claimed = [False]*nTrail
            for iTrail,fTrail in enumerate(foundTrails):

                widths[candidate.kind].append(fTrail.width)
                
                nMasked[candidate.kind] += fTrail.nMaskedPixels
                nMasked['all'] += fTrail.nMaskedPixels

                if t and  (candidate.kind in candi.Candidate.positiveKinds) and \
                   (fTrail.isNear(t, rMax, thetaMax)):
                    
                    resultMsg += "\n  %s: %s (%s)" % (clr.color("TRUE-POS", "green"), fTrail, candidate.kind)
                    claimed[iTrail] = True
                    eList.append(Event(True, True))

            ##########################################
            # False positives
            ##########################################
            nUnclaimed = len(claimed) - sum(claimed)
            if nUnclaimed > 0:
                nIgnored = 0
                resultTmp = ""

                for iClaim,claim in enumerate(claimed):
                    if not claim:
                        isIgnored = t and (candidate.kind in candi.Candidate.ignoredKinds) and \
                                    (foundTrails[iClaim].isNear(t, rMax, thetaMax))
                        isNegative = t and (candidate.kind in candi.Candidate.negativeKinds) and \
                                     (foundTrails[iClaim].isNear(t, rMax, thetaMax))
                        isOtherCandidate = False
                        for cand in candidateLookup[dataHash]:
                            if cand == candidate:
                                continue
                            if cand.trail and foundTrails[iClaim].isNear(cand.trail, rMax,thetaMax):
                                isOtherCandidate=True
                        if isNegative:
                            tag = "Known-bad"
                        elif isIgnored:
                            tag = "Ignored"
                            nIgnored += 1
                            resultMsg += "\n  %s: %s (%s)" % (clr.color("IGNORED-POS", "yellow"),
                                                              fTrail, candidate.kind)
                            continue
                        elif isOtherCandidate:
                            nIgnored += 1
                            continue
                        else:
                            tag = "Unclaimed"
                        resultTmp  += "  --> %s: %s\n" % (tag, clr.color(str(foundTrails[iClaim]), 'yellow'))
                        eList.append(Event(False, True))
                        falsePos.append((dataHash, foundTrails[iClaim]))
                        
                if nUnclaimed > nIgnored:
                    resultMsg += "\n  %s: %d Unclaimed trails (total=%d) (%s).\n" % \
                              (clr.color("FALSE-POS", "red"), nUnclaimed, nTrail, candidate.kind)
                    resultMsg += resultTmp

                
            #########################################
            # True negatives  and False ones
            #########################################
            if not resultMsg:
                # result does not match ... and it shouldn't
                if t is None  or (candidate.kind in \
                                  (candi.Candidate.negativeKinds | candi.Candidate.ignoredKinds)):
                    resultMsg += "\n  %s: %s (%s)" % (clr.color("TRUE-NEG", "green"),
                                                      "No trails present, and none found.", candidate.kind)
                    eList.append(Event(False, False))
                    
                # result does not match ... but it should have
                else:
                    resultMsg += "\n  %s: %s (%s)" % (clr.color("FALSE-NEG", "red"), t, candidate.kind)
                    eList.append(Event(True, False))
                    falseNeg.append((dataHash, t, candidate.kind))

                    
            did = "(%d, %d)" % (candidate.visit, candidate.ccd)
            if t:
                msg = "%s [r=%.1f theta=%.3f wid=%.1f]." % (clr.color(did, "cyan"), t.r, t.theta, t.width)
            else:
                msg = "%s [empty candidate]" % (clr.color(did, "cyan"))
            msg += resultMsg
            allMessages += msg+"\n"
            print "\n"+msg+"\n"
            eventLists[candidate.kind] += eList
            if candidate.kind not in candi.Candidate.ignoredKinds:
                eventLists['all'] += eList
        
    print clr.color("=== Summary ===", "magenta")
    print allMessages

    for kind, eventList in eventLists.items():
        print clr.color("=== %s ===" % (kind), "magenta")
        if len(eventList) == 0:
            print "No events."
            continue
        nPos = eventList.positiveDetections or 1
        print "TP=%d, TN=%d, FP=%d, FN=%d" % (eventList.truePositives, eventList.trueNegatives,
                                              eventList.falsePositives, eventList.falseNegatives)
        print "Recall,precision,f1:  %4.2f %4.2f  %4.2f"%(eventList.recall, eventList.precision, eventList.f1)
        print "Masked pixels: %10d" % (nMasked[kind]), \
            " (all: n = %d, %.2f%%)" % (len(eventList),100.0*nMasked[kind]/(len(eventList)*2048*4096)), \
            " (det: n = %d, %.2f%%)" % (eventList.positiveDetections,100.0*nMasked[kind]/(nPos*2048*4096))

    rt = np.array(runtimes)
    print "Runtimes:   mean=%.2f  med=%.2f  std=%.2f  min=%.2f  max=%.2f\n" % \
        (rt.mean(), np.median(rt), rt.std(), rt.min(), rt.max())

    for k,v in widths.items():
        print "Widths: %16s mean=%6.2f med=%6.2f std=%6.2f min=%6.2f max=%6.2f" % \
            (k, np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v))
    
    with open("falsePositives.txt", 'w') as fp:
        for d, f in falsePos:
            fp.write("%s %s\n" % (str(d), str(f)))
    with open("falseNegatives.txt", 'w') as fp:
        for d, f, k in falseNeg:
            fp.write("%s %s %s\n" % (str(d), k, str(f)))
                     
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("-c", "--candidates", choices=("all", "short"), default='all',
                        help="Candidate set to use.")
    parser.add_argument("-j", "--threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("-k", "--kind", default=None, help="Specify kind to run.")
    parser.add_argument("-v", "--visit", default=None, help="Specify visit to run")
    parser.add_argument("-s", "--sensor", default=None, help="Specify sensor (ccd) to run")
    parser.add_argument("-o", "--output", default="known.pickle")
    parser.add_argument("-i", "--input", default=None)
    args = parser.parse_args()

    if args.kind:
        args.kind = args.kind.split("^")
    if args.visit:
        args.visit = [int(x) for x in args.visit.split("^")]
    if args.sensor:
        args.sensor = [int(x) for x in args.sensor.split("^")]
    if args.output == 'None':
        args.output = None
        
    main(args.root, args.threads, args.output, args.input, kind=args.kind, visit=args.visit,
         sensor=args.sensor, candidateSet=candidateSets[args.candidates])
