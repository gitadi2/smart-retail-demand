// Vector similarity index for product cold-start neighbour search.
//
// Why C++: tight contiguous memory, SIMD dot products, predictable latency, and
// a zero-dependency C ABI callable from Python via ctypes (no pybind11 needed).
// Scope is deliberately focused -- a SIMD brute-force cosine index. For >~1M
// vectors you'd add IVF (k-means cells) or HNSW; the hooks are noted below.
//
// Vectors are L2-normalised on insert, so cosine similarity == dot product.
// Search returns the top-k by descending similarity.
//
// Build: see build.sh  ->  libvecindex.so

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

struct Index {
    int dim = 0;
    std::vector<float> data;       // n * dim, row-major, L2-normalised
    std::vector<int64_t> ids;      // length n
    int size() const { return static_cast<int>(ids.size()); }
};

// ---- dot product: AVX2 path + scalar fallback --------------------------------
static float dot(const float* a, const float* b, int dim) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    float s = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    for (; i < dim; ++i) s += a[i] * b[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
#endif
}

static void l2_normalise(float* v, int dim) {
    float n = std::sqrt(dot(v, v, dim));
    if (n > 1e-12f) for (int i = 0; i < dim; ++i) v[i] /= n;
}

extern "C" {

Index* vi_create(int dim) {
    Index* idx = new Index();
    idx->dim = dim;
    return idx;
}

void vi_free(Index* idx) { delete idx; }

int vi_size(Index* idx) { return idx ? idx->size() : 0; }

int vi_dim(Index* idx) { return idx ? idx->dim : 0; }

// Append `count` rows. Each row is normalised in place into the store.
void vi_add_batch(Index* idx, const int64_t* ids, const float* vecs, int count) {
    int d = idx->dim;
    size_t base = idx->data.size();
    idx->data.resize(base + static_cast<size_t>(count) * d);
    idx->ids.reserve(idx->ids.size() + count);
    for (int r = 0; r < count; ++r) {
        float* dst = idx->data.data() + base + static_cast<size_t>(r) * d;
        std::memcpy(dst, vecs + static_cast<size_t>(r) * d, sizeof(float) * d);
        l2_normalise(dst, d);
        idx->ids.push_back(ids[r]);
    }
}

// Top-k cosine search. Writes k ids + k similarities (descending).
// Unused slots (if n < k) are filled with id = -1, sim = -1.
void vi_search(Index* idx, const float* query, int k,
               int64_t* out_ids, float* out_sims) {
    int d = idx->dim, n = idx->size();
    std::vector<float> q(query, query + d);
    l2_normalise(q.data(), d);

    std::vector<float> sims(n);
    for (int i = 0; i < n; ++i)
        sims[i] = dot(q.data(), idx->data.data() + static_cast<size_t>(i) * d, d);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    int kk = std::min(k, n);
    std::partial_sort(order.begin(), order.begin() + kk, order.end(),
                      [&](int a, int b) { return sims[a] > sims[b]; });

    for (int i = 0; i < k; ++i) {
        if (i < kk) { out_ids[i] = idx->ids[order[i]]; out_sims[i] = sims[order[i]]; }
        else        { out_ids[i] = -1;                 out_sims[i] = -1.f; }
    }
}

// Persist as a flat binary blob (magic, dim, n, ids, data).
// Production note: mmap this file for zero-copy load of large indexes.
int vi_save(Index* idx, const char* path) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return -1;
    int32_t magic = 0x56494458; // "VIDX"
    int32_t dim = idx->dim, n = idx->size();
    std::fwrite(&magic, sizeof(magic), 1, f);
    std::fwrite(&dim, sizeof(dim), 1, f);
    std::fwrite(&n, sizeof(n), 1, f);
    std::fwrite(idx->ids.data(), sizeof(int64_t), n, f);
    std::fwrite(idx->data.data(), sizeof(float), static_cast<size_t>(n) * dim, f);
    std::fclose(f);
    return 0;
}

Index* vi_load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return nullptr;
    int32_t magic, dim, n;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x56494458) {
        std::fclose(f); return nullptr;
    }
    std::fread(&dim, sizeof(dim), 1, f);
    std::fread(&n, sizeof(n), 1, f);
    Index* idx = new Index();
    idx->dim = dim;
    idx->ids.resize(n);
    idx->data.resize(static_cast<size_t>(n) * dim);
    std::fread(idx->ids.data(), sizeof(int64_t), n, f);
    std::fread(idx->data.data(), sizeof(float), static_cast<size_t>(n) * dim, f);
    std::fclose(f);
    return idx;
}

} // extern "C"
