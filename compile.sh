#!/bin/bash

cd src
g++ -std=c++11 main.cpp -I . -I $CUDA_HOME/include -L $CUDA_HOME/lib64 -lcudart && ./a.out
