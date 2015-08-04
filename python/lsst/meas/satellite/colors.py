#!/usr/bin/env python

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
    """Wrap ascii text to show in color in terminal.

    @param text    The text you want colored
    @param color   The color (red,green,yellow,blue,magenta,cyan,grey)
    @param bold    Do you want bold-face font?

    @return string A string you can print which will show up in color.
    """
    
    base = "\033["
    code = colors[color]
    if bold:
        code += ";1"

    prefix = base + code + "m"
    suffix = base + "0m"
    return prefix + text + suffix

