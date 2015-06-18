#!/usr/bin/env bash

VISITS=270^1236^1168^242^1166^1184^1166^1166
CCDS=78^65^47^95^96^78^65^70

if [ "$1" = 'run' ]; then
    ./specific.py /lustre/Subaru/SSP/rerun/bick/cosmos333/ --id visit=$VISITS ccd=$CCDS -j 6
else
    echo "Not running.  Specify 'run' to reprocess."
fi

echo -n "See PNGs? [y/N]: "
read REPLY

if [ $REPLY = 'y' -o $REPLY = 'Y' ]; then

    DIR=data
    gm display $DIR/0270/satdebug-00270-078-b02.png &
    gm display $DIR/1236/satdebug-01236-065-b02.png &
    gm display $DIR/1168/satdebug-01168-047-b02.png &
    gm display $DIR/0242/satdebug-00242-095-b02.png &
    gm display $DIR/1166/satdebug-01166-096-b02.png &
    gm display $DIR/1184/satdebug-01184-078-b02.png &

    gm display $DIR/1166/satdebug-01166-065-b04.png &
    gm display $DIR/1166/satdebug-01166-070-b04.png &
    
else
    echo "Exiting."
fi



