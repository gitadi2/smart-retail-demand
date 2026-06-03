#!/usr/bin/env bash
# Build the vector index shared library callable from Python via ctypes.
# -march=native enables AVX2/AVX512 if the build host supports it.
set -e
cd "$(dirname "$0")"
g++ -O3 -march=native -fPIC -shared -std=c++17 index.cpp -o libvecindex.so
echo "built: $(pwd)/libvecindex.so"
