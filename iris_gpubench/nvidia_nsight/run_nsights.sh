#!/bin/bash
# Run profiling for given python script

# $1 is a string of the benchmark that is selected eg. "synthetic regresion

nsys profile ./benchmark.sh $1
