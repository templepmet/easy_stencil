#include <grid_series.hpp>
#include <type.hpp>

#include <cstdio>

int main(int argc, char **argv) {
    ull width = 1024;
    ull height = 1024;
    ull halo = 1;

    // set initial value by user
    TYPE *buf = new TYPE[width * height];

    ull size_mb = width * height * sizeof(TYPE) / 1e6;
    printf("width=%lld,height=%lld,size[MB]=%lld\n", width, height, size_mb);

    GridSeries<TYPE> grid_series(buf, width, height, halo);
    std::vector<ChunkHost<TYPE> *> &h_chunk_list =
        grid_series.getChunkHostList();
    std::vector<ChunkDevice<TYPE> *> &d_chunk_works =
        grid_series.getChunkDeviceWorks();
    assert(d_chunk_works.size() == 2);
    int chunkDim = h_chunk_list.size();

    printf("chunk_dim=%d\n", h_chunk_list.size());
    for (Chunk<TYPE> *chunk : h_chunk_list) {
        printf("width=%lld,height=%lld\n", chunk->getWidth(),
               chunk->getHeight());
    }

    int loops = 10;
    for (int i = 0; i < loops; ++i) {
        printf("loop=%d\n", i);
        for (int chunkIdx = 0; chunkIdx < chunkDim; ++chunkIdx) {
            // printf("chunkIdx=%d\n", chunkIdx);
            int step0 = chunkIdx % 2;
            int step1 = (chunkIdx + 1) % 2;
            ChunkHost<TYPE> *chunk = h_chunk_list[chunkIdx];
            ull chunk_height = chunk->getHeight();
            ull copy_rows = chunk_height + (chunkIdx < chunkDim - 1 ? halo : 0);
            d_chunk_works[step0]->Rows(0, copy_rows) =
                chunk->Rows(0, copy_rows);
            d_chunk_works[step1]->Rows(-halo + 1, halo) =
                d_chunk_works[step0]->Rows(chunk_height - halo, halo);
            d_chunk_works[step0]->compute();
            chunk->Rows(0, chunk_height) =
                d_chunk_works[step0]->Rows(0, chunk_height);
        }
    }
    grid_series.synchronize();

    delete[] buf;

    return 0;
}
