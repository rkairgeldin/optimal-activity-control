#!/bin/bash



python algorithm_3.py \
    -sr 5\
    -run 1\
    -T 5 \
    --verbose 1 \
    ER \
    -n 50 \
    -p 0.05 \
    $@
