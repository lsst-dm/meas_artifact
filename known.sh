#!/usr/bin/env bash

VISITS=270^1236^1168^242^1166^1184
AC_VISITS=1166^1166^1188^1240^1248
CCDS=78^65^47^95^96^78
AC_CCDS=65^70^18^51^43

if [ "$1" = 'run' -o "$1" = "rsat" ]; then
    ./specific.py /lustre/Subaru/SSP/rerun/bick/cosmos333/ --id visit=$VISITS ccd=$CCDS -j 6
else
    echo "Not running satellites.  Specify 'run' or 'sat' to reprocess."
fi

if [ "$1" = 'run' -o "$1" = "rac" ]; then
    ./specific.py /lustre/Subaru/SSP/rerun/bick/cosmos333/ --id visit=$AC_VISITS ccd=$AC_CCDS -j 5
else
    echo "Not running aircraft.  Specify 'run' or 'ac' to reprocess."
fi


# faint candidates
# 0262-015


echo -n "See PNGs? [y/N]: "
read REPLY

if [ $REPLY = 'y' -o $REPLY = 'Y' ]; then

    DIR=data

    if [ "$1" = 'run' -o "$1" = 'rsat' -o "$1" = 'sat' ]; then

        set -x
        cat $DIR/0270/log00270-078.txt
        cat $DIR/1236/log01236-065.txt
        cat $DIR/1168/log01168-047.txt
        cat $DIR/0242/log00242-095.txt
        cat $DIR/1166/log01166-096.txt
        cat $DIR/1184/log01184-078.txt
        set +x
        
        gm display $DIR/0270/satdebug-00270-078-SAT.png &
        gm display $DIR/0270/satdebug-00270-078-AC.png &
        gm display $DIR/1236/satdebug-01236-065-SAT.png &
        gm display $DIR/1236/satdebug-01236-065-AC.png &
        gm display $DIR/1168/satdebug-01168-047-SAT.png &
        gm display $DIR/1168/satdebug-01168-047-AC.png &
        gm display $DIR/0242/satdebug-00242-095-SAT.png &
        gm display $DIR/0242/satdebug-00242-095-AC.png &
        gm display $DIR/1166/satdebug-01166-096-SAT.png &
        gm display $DIR/1166/satdebug-01166-096-AC.png &
        gm display $DIR/1184/satdebug-01184-078-SAT.png &
        gm display $DIR/1184/satdebug-01184-078-AC.png &
    fi

    if [ "$1" = 'run' -o "$1" = "rac" -o "$1" = 'ac' ]; then

        cat $DIR/1166/log01166-065.txt
        cat $DIR/1166/log01166-070.txt   
        cat $DIR/1188/log01188-018.txt
        cat $DIR/1240/log01240-051.txt
        cat $DIR/1248/log01248-043.txt    
        
        gm display $DIR/1166/satdebug-01166-065-AC.png &
        gm display $DIR/1166/satdebug-01166-070-AC.png &
        gm display $DIR/1188/satdebug-01188-018-AC.png &
        gm display $DIR/1240/satdebug-01240-051-AC.png &
        gm display $DIR/1248/satdebug-01248-043-AC.png &
    fi
    
    
else
    echo "Exiting."
fi



