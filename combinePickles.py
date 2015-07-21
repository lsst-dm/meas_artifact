#!/usr/bin/env python
import sys, os, re
import cPickle as pickle
import argparse

def main(inputs, output):

    trails = []
    for input in inputs:
        with open(input, 'r') as fp:
            trails.append(pickle.load(fp))

    for bundle in trails:
        (v,c), trailList, runtime = bundle
        print v, c, trailList, runtime

            
    with open(output, 'w') as fp:
        pickle.dump(trails, fp)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="results.pickle", help="output filename")
    parser.add_argument("inputs", nargs="+", help="Input pickel files.")
    args = parser.parse_args()
    main(args.inputs, args.output)
