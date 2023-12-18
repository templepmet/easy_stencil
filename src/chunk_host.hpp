#pragma once

#include <chunk.hpp>
#include <type.hpp>
#include <handles.hpp>

#include <cuda.h>

#include <cassert>
#include <cstring>

template <class T> class ChunkDevice;

template <class T>
class ChunkHost : public Chunk<T> {
   private:

   public:
    ChunkHost(T *buf, ull width, ull height, ull halo)
        : Chunk<T>(buf, width, height, halo) {}
    
    // T& operator[](ull i) { return this->_buf[i]; }

    ChunkHost<T>& operator=(const ChunkDevice<T>& right_chunk)
    {
        assert(this->_target_size == right_chunk.getTargetSize());
        HANDLE_CUDA(cudaMemcpyAsync(this->_target_ptr, right_chunk.getTargetPtr(), this->_target_size, cudaMemcpyDeviceToHost));
        return *this;
    }

    ChunkHost<T>& operator=(const ChunkHost<int>& right_chunk)
    {
        assert(this->_target_size == right_chunk.getTargetSize());
        memcpy(this->_target_ptr, right_chunk.getTargetPtr(), this->_target_size);
        return *this;
    }

    ChunkHost<T>& Rows(ull start_row, ull rows) {
        this->_target_ptr = this->_buf + this->_width * start_row;
        this->_target_size = sizeof(T) * this->_width * rows;
        return *this;
    }
    ChunkHost<T>& Rows() { return Rows(0, this->_height); }
};
