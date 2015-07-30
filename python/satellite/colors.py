#!/usr/bin/env python

import os, sys
import argparse

"""
%prog [options] text
"""

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

def main(text, clr, bold=False):

    if clr == 'all':
        for c, n in sorted(colors.items(), key=lambda x: x[1]):
            print n, ": ", color(text, c, bold=bold)
    else:
        print color(text, clr, bold)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to colorize")
    parser.add_argument("-c", "--color", help="Specify color: "+",".join(colors.keys()), required=False, default='red')
    parser.add_argument("-b", "--bold", help="Make bold", required=False, default=False, action='store_true')

    args = parser.parse_args()
    
    main(args.text, clr=args.color, bold=args.bold)
