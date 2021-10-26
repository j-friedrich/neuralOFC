#!/bin/bash

for loop in closed open
do
    for delay in 1 2 3
    do
        # main figures
        for s in 0 1 2
        do
            python hyperopt.py $s $delay $loop .2 .99
        done
        # supplementary figures
        for sigma in .05 .1 .5
        do
            python hyperopt.py 0 $delay $loop $sigma .99
        done
        for momentum in 0 .9 .9995
        do
            python hyperopt.py 0 $delay $loop .2 $momentum
        done        
    done
done
