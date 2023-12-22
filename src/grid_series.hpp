#pragma once

#include <chunk_device.hpp>
#include <chunk_host.hpp>
#include <handles.hpp>
#include <type.hpp>

#include <cuda.h>

#include <vector>

template <class T>
class GridSeries {
   private:
    std::vector<ChunkHost<T>*> _h_chunk_list;
    std::vector<ChunkDevice<T>*> _d_chunk_works;

    T* _h_buf;
    ull _width;
    ull _height;
    ull _halo;

   public:
    GridSeries(T* h_buf, ull width, ull height, ull halo)
        : _h_buf(h_buf), _width(width), _height(height), _halo(halo) {
        // split buf to chunk
        size_t available_size;
        size_t total_size;
        HANDLE_CUDA(cudaMemGetInfo(&available_size, &total_size));
        available_size = 256 * 256 * sizeof(int);  // debug

        // currently split in row major
        // maximize x {(bs * 2 + x) * width * 2 < available_size && x <= height}
        ull d_work_height = std::min(
            (available_size + (_width * 2 - 1)) / (_width * 2) - _halo * 2,
            height);
        printf("d_work_height=%lld\n", d_work_height);
        ull d_work_size = sizeof(T) * width * (d_work_height + halo * 2);
        _d_chunk_works.resize(2);  // double buffering
        _d_chunk_works[0] = new ChunkDevice<T>(width, d_work_height, halo);
        _d_chunk_works[1] = new ChunkDevice<T>(width, d_work_height, halo);

        T* h_start_buf = _h_buf;
        ull start_row = 0;
        while (start_row < height) {
            ull chunk_height = std::min(d_work_height, height - start_row);
            _h_chunk_list.push_back(
                new ChunkHost<T>(h_start_buf, width, chunk_height, halo));
            h_start_buf += width * chunk_height;
            start_row += chunk_height;
        }
    }

    ~GridSeries() {
        for (Chunk<T>* chunk : _h_chunk_list) {
            delete chunk;
        }
        delete _d_chunk_works[0];
        delete _d_chunk_works[1];
    }

    std::vector<ChunkHost<T>*>& getChunkHostList() { return _h_chunk_list; }
    std::vector<ChunkDevice<T>*>& getChunkDeviceWorks() {
        return _d_chunk_works;
    }

    void synchronize() {
        HANDLE_CUDA(cudaStreamSynchronize(0));  // should be all used stream
    }
};
