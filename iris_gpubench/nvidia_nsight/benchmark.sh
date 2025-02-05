#!/bin/bash

# $1 is a string of the benchmark that is selected eg. "synthetic regresion"

docker run -d --gpus all --shm-size=1024g $1
