#!/bin/bash

# Loop from 1 to 30 for the seed value
for i in {1..30}
do
    echo "Running simulation with seed $i"
    mpirun -np 8 python stokes.py $i
done
