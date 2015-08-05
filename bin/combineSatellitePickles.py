#!/usr/bin/env python
import sys, os, re
import cPickle as pickle
import argparse

def main(inputs, output, verbose=False):

    trails = []
    for input in inputs:
        print "loading", input
        with open(input, 'r') as fp:
            trails.append(pickle.load(fp))
    if output:
        with open(output, 'w') as fp:
            pickle.dump(trails, fp)
            
    if verbose:
        for bundle in trails:
            (v,c), trailList, runtime = bundle
            print v, c, trailList, runtime
            for t in trailList:
                print "    ", t

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input pickel files.")
    parser.add_argument("-o", "--output", help="output filename")
    parser.add_argument("-v", "--verbose", type=bool, default=False, action='set_true',
                        help="Print the contents to stdout")
    args = parser.parse_args()
    main(args.inputs, args.output, verbose=args.verbose)
