#pragma once

#include <type.hpp>

template <class T>
class Chunk {
   protected:
    T* _buf;
    ull _width;
    ull _height;
    ull _halo;
    
    T* _target_ptr;
    ull _target_size;

   public:
    Chunk(T* buf, ull width, ull height, ull halo)
        : _buf(buf), _width(width), _height(height), _halo(halo), _target_ptr(nullptr), _target_size(0) {}

    T* getPtr() const { return _buf; }
    ull getWidth() const { return _width; }
    ull getHeight() const { return _height; }
    ull getHalo() const { return _halo; }
    T* getTargetPtr() const { return _target_ptr; }
    ull getTargetSize() const { return _target_size; }
};
