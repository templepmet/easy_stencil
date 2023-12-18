#pragma once

#include <chunk.hpp>
#include <type.hpp>
#include <handles.hpp>

#include <cuda.h>

#include <cassert>

template <class T> class ChunkHost;

template <class T>
class ChunkDevice : public Chunk<T> {
   private:

   public:
    ChunkDevice(ull width, ull height, ull halo)
        : Chunk<T>(nullptr, width, height, halo) {
        HANDLE_CUDA(cudaMalloc(&this->_buf, sizeof(T) * this->_width * (this->_height + this->_halo * 2)));
    }

    ~ChunkDevice() {
        HANDLE_CUDA(cudaFree(this->_buf));
    }


    ChunkDevice<T>& operator=(const ChunkDevice<T>& right_chunk)
    {
        assert(this->_target_size == right_chunk.getTargetSize());
        HANDLE_CUDA(cudaMemcpyAsync(this->_target_ptr, right_chunk.getTargetPtr(), this->_target_size, cudaMemcpyDeviceToDevice));
        return *this;
    }

    ChunkDevice<T>& operator=(const ChunkHost<int>& right_chunk)
    {
        assert(this->_target_size == right_chunk.getTargetSize());
        HANDLE_CUDA(cudaMemcpyAsync(this->_target_ptr, right_chunk.getTargetPtr(), this->_target_size, cudaMemcpyHostToDevice));
        return *this;
    }

    ChunkDevice<T>& Rows(ull start_row, ull rows) {
        this->_target_ptr = this->_buf + this->_width * start_row;
        this->_target_size = sizeof(T) * this->_width * rows;
        return *this;
    }
    ChunkDevice<T>& Rows() { return Rows(0, this->_height); }

    void compute(ull start_row, ull rows) {}
    void compute() {}
};
