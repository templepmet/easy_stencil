#!/bin/bash -eu

g++ -std=c++11 -g -O3 main.cpp -I src -I $CUDA_HOME/include -L $CUDA_HOME/lib64 -lcudart
./a.out
