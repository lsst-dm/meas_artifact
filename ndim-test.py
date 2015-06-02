#!/usr/bin/env python

import sys, os, re, argparse
import numpy as np
from scipy import ndimage as ndimg

def main():

    nx, ny = 16,16
    data = np.zeros((nx, ny)).astype(int)

    n_groups = 3
    s = 1
    xx = np.random.uniform(s, nx-s, size=n_groups).astype(int)
    yy = np.random.uniform(s, ny-s, size=n_groups).astype(int)

    value = 1
    for i in range(n_groups):
        x = xx[i]
        y = yy[i]
        print x, y
        data[y-s:y+s,x-s:x+s] += value
        value += 1
    
    locus, numLocus = ndimg.label(data > 1)

    for i in range(numLocus):

        wy,wx = np.where(data == i+1)

        print data[wy,wx]

    print data
    print numLocus
    print locus

if __name__ == '__main__':
    main()
