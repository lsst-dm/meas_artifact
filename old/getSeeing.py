#!/usr/bin/env python

import argparse
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as ellipse
import lsst.daf.persistence  as dafPersist


def main(root, tract, patch, filter):

    # make a butler and specify your dataId
    butler = dafPersist.Butler(root)
    dataId = {'tract':tract, 'patch':patch, 'filter':filter}

    # get the exposure from the butler
    exposure = butler.get('deepCoadd', dataId)

    # get the PSF in the center of the image
    x0, y0 = exposure.getX0(), exposure.getY0()
    midpixel = afwGeom.Point2D(x0+exposure.getWidth()//2, y0+exposure.getHeight()//2)
    psfshape = exposure.getPsf().computeShape(midpixel)
    psfaxes = ellipse.Axes(psfshape)
    
    # use semi-major axis as an estimate of PSF width
    psfwidth = 2.3548*(psfaxes.getA()*psfaxes.getB())**0.5
    pixelScale = 0.168
    print tract, patch, filter, psfwidth*pixelScale

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of data repository")
    parser.add_argument("tract", type=int, help="Tract to show")
    parser.add_argument("patch", help="Patch to show")
    parser.add_argument("filter", help="Filter to show")
    args = parser.parse_args()

    main(args.root, args.tract, args.patch, args.filter)
