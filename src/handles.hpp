#pragma once

#include <cuda_runtime.h>

#include <cstdio>

#define HANDLE_CUDA(call)                                                      \
    {                                                                          \
        cudaError_t cudaStatus = call;                                         \
        if (cudaStatus != cudaSuccess) {                                       \
            fprintf(stderr,                                                    \
                    "ERROR: HANDLE_CUDA \"%s\" in line %d of file %s failed "  \
                    "with "                                                    \
                    "%s (%d).\n",                                              \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), \
                    cudaStatus);                                               \
            exit(cudaStatus);                                                  \
        }                                                                      \
    }
