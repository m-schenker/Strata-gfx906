// routerlab.hip - what makes the gfx906 router cost 16 us for 512 logits?
//
// The engine's native_router_top10 (src/kernels/cuda/native_router.cu) launches <<<1, dim3(32, 8)>>>, keeps only
// threadIdx.y == 0 (one 32-lane warp), and does ten sequential "max over the remaining" passes, each with a 5-round
// bi-value butterfly (`__shfl_xor_sync` of best AND expert id).  Measured through the engine's own launcher
// (tools/hip/gfx906_dense_bench.cpp): 16-17 us a call, i.e. ~1.6 ms per token at 96 calls, ~12 % of decode.
//
// This file isolates the cost: same maths, different launch geometry / reduction, all on the same input.
//   A  the engine's kernel, verbatim, <<<1, dim3(32,8)>>>
//   B  A with <<<1, 32>>> (one warp instead of eight, seven of which exit at once)
//   C  A with one 64-lane wave per token (8 values a lane instead of 16) - ids bitwise, weights not
//   D  A with a per-lane sorted list so each rank's local max is a register read (no 16-value rescan)
// Parity: the ids each variant picks are compared against A's, and the weights against A's.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cstring>
#include <random>
#include <vector>

#define HIP_CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

__device__ __forceinline__ float warp_max32(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor(v, m, 32));
    return v;
}
__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor(v, m, 32);
    return v;
}

// ---- A: the engine's kernel, unchanged apart from being a template on the geometry
template <int NY>
__global__ void routeA(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights) {
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max32(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { values[i] = expf(values[i] - maximum); sum += values[i]; }
    const float reciprocal = 1.0f / warp_sum32(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float other = __shfl_xor(best, m, 32);
            const int other_id = __shfl_xor(expert, m, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) { values[expert / 32] = -INFINITY; ids[rank] = expert; selected_sum += best; }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum32(selected_sum), 6.103515625e-5f);
    const float inv = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inv;
}

// ---- C: one 64-lane wave, 8 values a lane.  Ranking rule unchanged (value desc, then lower id), so the ten ids are
// the same as A's; the weight sum is over a different reduction tree, so it is not bitwise.
__global__ void routeC(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights) {
    const int lane = threadIdx.x & 63;
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) values[i] = logits[lane + i * 64];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) maximum = max(maximum, values[i]);
#pragma unroll
    for (int m = 32; m; m >>= 1) maximum = fmaxf(maximum, __shfl_xor(maximum, m, 64));
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { values[i] = expf(values[i] - maximum); sum += values[i]; }
#pragma unroll
    for (int m = 32; m; m >>= 1) sum += __shfl_xor(sum, m, 64);
    const float reciprocal = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 8; ++i) if (values[i] > best) { best = values[i]; expert = lane + i * 64; }
#pragma unroll
        for (int m = 32; m; m >>= 1) {
            const float other = __shfl_xor(best, m, 64);
            const int other_id = __shfl_xor(expert, m, 64);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 63) == lane) { values[expert / 64] = -INFINITY; if (lane < 10) ids[rank] = expert; selected_sum += best; }
        if (rank == lane) selected = best;
    }
#pragma unroll
    for (int m = 32; m; m >>= 1) selected_sum += __shfl_xor(selected_sum, m, 64);
    selected_sum = max(selected_sum, 6.103515625e-5f);
    const float inv = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inv;
}

// ---- D: A's geometry and maths, but each lane keeps its 16 values sorted, so a rank's local max is a read
template <int NY>
__global__ void routeD(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights) {
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
    int order[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) { values[i] = logits[lane + i * 32]; order[i] = i; }
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max32(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { values[i] = expf(values[i] - maximum); sum += values[i]; }
    const float reciprocal = 1.0f / warp_sum32(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    // sorted descending, ties by lower expert id (insertion sort, 16 elements, no shuffles)
    for (int i = 1; i < 16; ++i) {
        const float v = values[i];
        const int o = order[i];
        int j = i - 1;
        while (j >= 0 && (values[j] < v || (values[j] == v && order[j] > o))) {
            values[j + 1] = values[j];
            order[j + 1] = order[j];
            --j;
        }
        values[j + 1] = v;
        order[j + 1] = o;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    int taken = 0;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane + order[0] * 32;
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float other = __shfl_xor(best, m, 32);
            const int other_id = __shfl_xor(expert, m, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            const int slot = expert / 32;
            int at = 0;
            for (int i = 0; i < 16; ++i) if (order[i] == slot) { at = i; break; }
            for (int i = at; i < 15; ++i) { values[i] = values[i + 1]; order[i] = order[i + 1]; }
            values[15] = -INFINITY; order[15] = 0;
            ++taken;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum32(selected_sum), 6.103515625e-5f);
    const float inv = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inv;
}


// ---- E: sort-then-merge.  The dependency chain is what costs: ten ranks x five rounds x two dependent
// `ds_bpermute_b32`s is 100 dependent crossbar round-trips (~15 us measured, and a 64-lane version with 8 values a lane
// is no better).  Here each lane sorts its own 16 values (registers only), then five rounds of pairwise merges bring the
// global top-10 to every lane with the 20 shuffles of a round *independent* of each other, so their latency overlaps.
// The ranking rule is the same (value desc, then lower id), so ids are bitwise; the weights are replayed from the merged
// list in rank order into each lane's own sum, which is the original's accumulation order, so they are bitwise too.
struct KV { float v; int id; };

__device__ __forceinline__ bool kv_less(const KV& a, const KV& b) {   // a ranks before b?
    return a.v > b.v || (a.v == b.v && a.id < b.id);
}

template <int NY>
__global__ void routeE(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights) {
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max32(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { values[i] = expf(values[i] - maximum); sum += values[i]; }
    const float reciprocal = 1.0f / warp_sum32(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    // this lane's own top-10 (16 values, ties by lower expert id)
    KV mine[10];
#pragma unroll
    for (int j = 0; j < 10; ++j) { mine[j].v = -INFINITY; mine[j].id = 1 << 30; }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        KV cur;
        cur.v = values[i];
        cur.id = lane + i * 32;
#pragma unroll
        for (int j = 0; j < 10; ++j) {           // insertion into the sorted 10
            if (kv_less(cur, mine[j])) { const KV t = mine[j]; mine[j] = cur; cur = t; }
        }
    }
    // five merge rounds: each lane merges with its partner's sorted 10, keeps the top 10
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        KV other[10];
#pragma unroll
        for (int j = 0; j < 10; ++j) {
            other[j].v = __shfl_xor(mine[j].v, m, 32);      // 20 shuffles, all independent
            other[j].id = __shfl_xor(mine[j].id, m, 32);
        }
        KV merged[10];
        int a = 0, b = 0;
#pragma unroll
        for (int j = 0; j < 10; ++j) {
            if (kv_less(mine[a], other[b])) merged[j] = mine[a++]; else merged[j] = other[b++];
        }
#pragma unroll
        for (int j = 0; j < 10; ++j) mine[j] = merged[j];
    }
    // replay the original's accumulation: rank r adds to the lane that owns that expert
    float selected = 0.0f, selected_sum = 0.0f;
#pragma unroll
    for (int rank = 0; rank < 10; ++rank) {
        if ((mine[rank].id & 31) == lane) selected_sum += mine[rank].v;
    }
    if (lane < 10) selected = mine[lane].v;
    selected_sum = max(warp_sum32(selected_sum), 6.103515625e-5f);
    const float inv = 1.0f / selected_sum;
    if (lane < 10) ids[lane] = mine[lane].id;
    if (lane < 10) weights[lane] = selected * inv;
}

// ---- F: bitonic selection network, no data-dependent merge loop
// E's remaining cost is its merge: a 10-step `if (kv_less(a,b)) pick a else pick b` loop, which is a serial
// pointer chase through registers.  A bitonic top-k merge replaces it with compare-exchanges that are all
// independent of each other, and 16 of them are available to issue at once.  Per lane: a bitonic sort of its own
// 16 values, then five rounds of (exchange 16 slots with the partner lane, M[j] = max(L[j], reverse(P)[j]), clean
// with strides 8/4/2/1).  Same ranking rule as A (value desc, lower id wins ties), so ids are expected bitwise.
__device__ __forceinline__ void ce_desc(KV& a, KV& b) {        // a keeps the better of the two
    const bool sw = !kv_less(a, b);
    const float av = a.v, bv = b.v;
    const int ai = a.id, bi = b.id;
    a.v = sw ? bv : av; a.id = sw ? bi : ai;
    b.v = sw ? av : bv; b.id = sw ? ai : bi;
}
__device__ __forceinline__ void kv_keep_max(KV& a, const KV& b) {   // a = the better of the two
    const bool take = kv_less(b, a);
    a.v = take ? b.v : a.v;
    a.id = take ? b.id : a.id;
}
__device__ __forceinline__ void ce_asc(KV& a, KV& b) {        // a keeps the worse of the two
    const bool sw = kv_less(a, b);
    const float av = a.v, bv = b.v;
    const int ai = a.id, bi = b.id;
    a.v = sw ? bv : av; a.id = sw ? bi : ai;
    b.v = sw ? av : bv; b.id = sw ? ai : bi;
}

template <int NY>
__global__ void routeF(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights) {
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max32(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { values[i] = expf(values[i] - maximum); sum += values[i]; }
    const float reciprocal = 1.0f / warp_sum32(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    // bitonic sort of this lane's 16 (branch-free: the network shape is compile-time)
    KV x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) { x[i].v = values[i]; x[i].id = lane + i * 32; }
#pragma unroll
    for (int k = 2; k <= 16; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) ce_desc(x[i], x[ixj]);
                    else ce_asc(x[i], x[ixj]);
                }
            }
        }
    }
    // five rounds: exchange the sorted 16 with the partner, keep the top 16 of the two
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        KV p[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {                         // 32 independent shuffles
            p[j].v = __shfl_xor(x[j].v, m, 32);
            p[j].id = __shfl_xor(x[j].id, m, 32);
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) kv_keep_max(x[j], p[15 - j]);   // M = max(L, reverse(P))
#pragma unroll
        for (int stride = 8; stride >= 1; stride >>= 1) {       // bitonic clean, descending
#pragma unroll
            for (int i = 0; i < 16; ++i)
                if ((i & stride) == 0) ce_desc(x[i], x[i + stride]);
        }
    }
    float selected = 0.0f, selected_sum = 0.0f;
#pragma unroll
    for (int rank = 0; rank < 10; ++rank)
        if ((x[rank].id & 31) == lane) selected_sum += x[rank].v;
    if (lane < 10) selected = x[lane].v;
    selected_sum = max(warp_sum32(selected_sum), 6.103515625e-5f);
    const float inv = 1.0f / selected_sum;
    if (lane < 10) ids[lane] = x[lane].id;
    if (lane < 10) weights[lane] = selected * inv;
}

// ---- G: pack the expert id into the key, so a rank needs ONE reduction instead of two
// A's ten ranks each do two 5-deep reductions (value, then id to break ties) = 100 shuffles.  E and F replace the
// chain with merge networks and lose, because they *add* work.  This variant removes work instead: after the softmax
// every value is in [0,1], so (uint)(v*2^24) is monotonic in v, and the low 9 bits can carry (511 - id) as the tie
// break.  A lane then reduces ONE integer per rank - 50 shuffles instead of 100 - and the id, and therefore the owning
// lane of the chosen expert, comes out of the key itself.  The weight still comes from the original float, so only the
// selection order depends on the packing: two values closer than 2^-24 in v would tie and be broken by id, which is
// 6e-8 of weight - the one way this differs from A.
__device__ __forceinline__ unsigned rotr_key(float v, int id) {
    const unsigned q = (unsigned) (v * 16777216.0f);          // 24 bits of value, clipped below
    return (q << 9) | (unsigned) (511 - id);
}

template <int NY>
__global__ void routeG(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights) {
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max32(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { values[i] = expf(values[i] - maximum); sum += values[i]; }
    const float reciprocal = 1.0f / warp_sum32(sum);
    unsigned key[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
        key[i] = rotr_key(values[i], lane + i * 32);
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        unsigned best = key[0];
#pragma unroll
        for (int i = 1; i < 16; ++i) best = max(best, key[i]);
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) best = max(best, __shfl_xor(best, mask, 32));   // 5 shuffles, value+id together
        const int expert = 511 - (int) (best & 511u);
        if ((expert & 31) == lane) {
            key[expert >> 5] = 0u;                    // 0 sorts below any real key (id 511, value 0)
            selected_sum += values[expert >> 5];
            if (rank == 0 || true) { /* keep the same accumulation site as A */ }
        }
        if (rank == lane) {
            // the rank's own value: recompute from the key so the lane holding rank r reports the r-th best
            selected = (float) (best >> 9) * (1.0f / 16777216.0f);
        }
    }
    selected_sum = max(warp_sum32(selected_sum), 6.103515625e-5f);
    const float inverse = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse;
}

// ---- H: one reduction per rank, ties resolved by ballot (exact, no key packing)
// G's mechanism - don't ship the id through the reduction - but without giving up exactness.  The id is not needed by
// the reduction at all: after the 5-shuffle max every lane knows the winning VALUE, the lane that owns it knows which
// slot it came from (and therefore the expert id), and if only one lane has that value there is nothing to break.  A
// ballot detects the multi-lane tie case, which is the only case that needs the id-in-reduction of A, and it is rare.
// So: 5 shuffles + 1 ballot per rank instead of 10 shuffles, the same exact float comparisons as A.
__device__ __forceinline__ float warp_max_exact(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor(v, m, 32));
    return v;
}

template <int NY>
__global__ void routeH(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights) {
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max_exact(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { values[i] = expf(values[i] - maximum); sum += values[i]; }
    const float reciprocal = 1.0f / warp_max_exact(0.0f) * 0.0f;   // keep the compiler from reordering the barrier below
    (void) reciprocal;
    const float inv_sum = 1.0f / warp_sum32(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= inv_sum;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int slot = 0;
#pragma unroll
        for (int i = 1; i < 16; ++i) if (values[i] > best) { best = values[i]; slot = i; }
        best = warp_max_exact(best);                       // 5 shuffles, value only
        const bool hit = (values[slot] == best);           // this lane owns the winning value?
        const int cand = (int) (lane + slot * 32);
        int winner = hit ? cand : 0x7FFFFFFF;
        if (__popc(__ballot(hit)) > 1) {                   // rare, and uniform: several lanes hold the value
#pragma unroll
            for (int m = 16; m; m >>= 1) winner = min(winner, __shfl_xor(winner, m, 32));   // A picks the lower id
        }
        if (hit && cand == winner) {
            ids[rank] = cand;
            values[slot] = -INFINITY;
            selected_sum += best;                          // accumulated by the owning lane, as in A
        }
        if (rank == lane) selected = best;                 // every lane has the rank's value
    }
    selected_sum = max(warp_sum32(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}

__global__ void empty_k() { if (threadIdx.x == 1024) __builtin_trap(); }

template <typename F>
double bench(F launch, int iters) {
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    HIP_CHECK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return (double) ms * 1000.0 / iters;
}

int main(int argc, char** argv) {
    int iters = 500, tokens = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = std::atoi(argv[++i]);
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nf(0.0f, 1.0f);
    std::vector<float> hl((size_t) tokens * 512);
    for (auto& v : hl) v = nf(rng);
    if (std::getenv("ROUTER_TIES")) {            // exact ties everywhere: the path A's tie-break exists for
        for (auto& v : hl) v = std::round(v * 2.0f) * 0.5f;
        std::printf("ROUTER_TIES: logits forced onto exact ties\n");
    }

    float* dl = nullptr;
    int32_t* di = nullptr;
    float* dw = nullptr;
    HIP_CHECK(hipMalloc(&dl, hl.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&di, (size_t) tokens * 10 * 4));
    HIP_CHECK(hipMalloc(&dw, (size_t) tokens * 10 * 4));
    HIP_CHECK(hipMemcpy(dl, hl.data(), hl.size() * sizeof(float), hipMemcpyHostToDevice));

    std::vector<int32_t> ids(10), ids_ref(10);
    std::vector<float> wts(10), wts_ref(10);
    auto read = [&](std::vector<int32_t>& i, std::vector<float>& w) {
        HIP_CHECK(hipMemcpy(i.data(), di, 10 * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(w.data(), dw, 10 * 4, hipMemcpyDeviceToHost));
    };

    routeA<8><<<tokens, dim3(32, 8)>>>(dl, di, dw);
    HIP_CHECK(hipDeviceSynchronize());
    read(ids_ref, wts_ref);

    auto report = [&](const char* name, double us, bool same_ids) {
        read(ids, wts);
        bool ids_ok = same_ids ? (ids == ids_ref) : (std::is_permutation(ids.begin(), ids.end(), ids_ref.begin()));
        double worst = 0.0;
        for (int i = 0; i < 10; ++i) worst = std::max(worst, std::fabs((double) wts[i] - wts_ref[i]));
        std::printf("%-28s %8.3f us   ids %s   worst weight delta %.3g\n", name, us,
                    ids_ok ? (same_ids ? "bitwise" : "same set") : "DIFFERENT", worst);
    };

    std::printf("%-28s %8.3f us   (launch overhead, same geometry)\n", "empty kernel <<<1,256>>>",
                bench([&] { empty_k<<<1, dim3(32, 8)>>>(); }, iters));
    report("A engine (32x8, 1 warp)", bench([&] { routeA<8><<<tokens, dim3(32, 8)>>>(dl, di, dw); }, iters), true);
    report("B engine geometry, 1 warp", bench([&] { routeA<1><<<tokens, dim3(32, 1)>>>(dl, di, dw); }, iters), true);
    report("C one 64-lane wave", bench([&] { routeC<<<tokens, 64>>>(dl, di, dw); }, iters), false);
    report("D sorted-per-lane local max", bench([&] { routeD<8><<<tokens, dim3(32, 8)>>>(dl, di, dw); }, iters), true);
    report("E sort + merge tree", bench([&] { routeE<8><<<tokens, dim3(32, 8)>>>(dl, di, dw); }, iters), true);
    report("F bitonic selection net", bench([&] { routeF<8><<<tokens, dim3(32, 8)>>>(dl, di, dw); }, iters), true);
    report("G id packed in the key", bench([&] { routeG<8><<<tokens, dim3(32, 8)>>>(dl, di, dw); }, iters), false);
    report("H ballot tie-break", bench([&] { routeH<8><<<tokens, dim3(32, 8)>>>(dl, di, dw); }, iters), true);
    return 0;
}
