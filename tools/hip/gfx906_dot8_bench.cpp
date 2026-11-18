// tools/hip/gfx906_dot8_bench.cpp - does v_dot8_i32_i4 pay on this card, and what does it cost?
//
//     clang++ -x hip --offload-arch=gfx906 --rocm-path=/opt/rocm -O3 -o dot8 gfx906_dot8_bench.cpp && ./dot8
//
// The lever: v_dot4_i32_i8 (dp4a) consumes 4 int8 pairs per issue, v_dot8_i32_i4 consumes 8 int4 pairs per issue, so a
// 4-bit weight matrix needs half the dot instructions for the same MACs - twice the ALU throughput on a card that has
// fast HBM and no matrix cores.  Both instructions exist on gfx906 and clang emits them for it (checked in the ISA:
// v_dot4_i32_i8 / v_dot8_i32_i4 / v_dot8_u32_u4).
//
// The catch: dot8 needs BOTH operands in 4-bit fields.  A 4-bit weight matrix fits (Q4_0/Q4_1 store values in
// nibbles), but an engine's activations are q8_1 - 8-bit - so a dot8 path has to quantise the activations to 4 bits as
// well.  That is a numerics change, not a free speedup, so this tool measures both sides on the same data:
//
//   arm dp4a : 4-bit weights in Q4_0's offset-binary nibbles (value = nibble - 8, one v_sub per 8 values) x q8_1
//   arm dot8 : the same values in two's-complement nibbles (what the instruction reads) x 4-bit activations
//
// Same weight bytes streamed by both arms, so a time difference is issue/ALU, not memory.  Reported per arm: time,
// G MAC/s, and the error against an FP32 reference over the same weights and activations.
//
// Layout mirrors the engine's decode GEMV: one 64-lane wave per row, 32 values per block per lane step, wave butterfly.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP %s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {

constexpr int QK = 32;                      // values per block (the engine's q8_1 block)

struct alignas(16) BlockW {                 // 32 four-bit values in 4 u32 + one f32 scale
    float d;
    uint32_t q[QK / 8];
};
struct alignas(16) BlockQ8 {                // 8-bit activations (what the engine runs today)
    float d;
    int8_t qs[QK];
};
struct alignas(16) BlockQ4 {                // 4-bit activations (what dot8 needs)
    float d;
    uint32_t q[QK / 8];
};

// byte j of a u32 holds values (8i + j) in its low nibble and (8i + 4 + j) in its high nibble, matching Q4_0's
// low/high split so dp4a can take two int8 quads straight out of one word.
__global__ void gemv_dp4a(const BlockW* __restrict__ w, const BlockQ8* __restrict__ x, float* __restrict__ y,
                          int n_row, int n_blk) {
    const int lane = threadIdx.x & 63;
    const int row = blockIdx.x * (blockDim.x / 64) + (threadIdx.x >> 6);
    if (row >= n_row) return;
    float acc = 0.0f;
    for (int b = lane; b < n_blk; b += 64) {
        const BlockW* wb = w + (size_t) row * n_blk + b;
        const BlockQ8* xb = x + b;
        const int* x8 = (const int*) xb->qs;
        int s = 0;
#pragma unroll
        for (int i = 0; i < QK / 8; ++i) {
            const uint32_t v = wb->q[i];
            const uint32_t lo = (v & 0x0F0F0F0Fu) - 0x08080808u;          // offset binary -> int8 (Q4_0's trick)
            const uint32_t hi = ((v >> 4) & 0x0F0F0F0Fu) - 0x08080808u;
            s = __builtin_amdgcn_sdot4((int) lo, x8[2 * i], s, false);
            s = __builtin_amdgcn_sdot4((int) hi, x8[2 * i + 1], s, false);
        }
        acc += xb->d * wb->d * (float) s;
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 64);
    if (lane == 0) y[row] = acc;
}

__global__ void gemv_dot8(const BlockW* __restrict__ w, const BlockQ4* __restrict__ x, float* __restrict__ y,
                          int n_row, int n_blk) {
    const int lane = threadIdx.x & 63;
    const int row = blockIdx.x * (blockDim.x / 64) + (threadIdx.x >> 6);
    if (row >= n_row) return;
    float acc = 0.0f;
    for (int b = lane; b < n_blk; b += 64) {
        const BlockW* wb = w + (size_t) row * n_blk + b;
        const BlockQ4* xb = x + b;
        int s = 0;
#pragma unroll
        for (int i = 0; i < QK / 8; ++i) s = __builtin_amdgcn_sdot8((int) wb->q[i], (int) xb->q[i], s, false);
        acc += xb->d * wb->d * (float) s;
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 64);
    if (lane == 0) y[row] = acc;
}

struct Case { int rows, k, groups; };

double bench(bool dot8, const Case& c, void* w, void* x, float* y, int iters) {
    auto launch = [&](int reps) {
        const int wpb = 4;
        const unsigned grid = (unsigned) ((c.rows * c.groups + wpb - 1) / wpb);
        for (int i = 0; i < reps; ++i) {
            if (dot8)
                gemv_dot8<<<grid, 64 * wpb>>>((const BlockW*) w, (const BlockQ4*) x, y, c.rows * c.groups, c.k / QK);
            else
                gemv_dp4a<<<grid, 64 * wpb>>>((const BlockW*) w, (const BlockQ8*) x, y, c.rows * c.groups, c.k / QK);
        }
    };
    launch(3);
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    HIP_CHECK(hipEventRecord(a));
    launch(iters);
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return ms / iters;
}

}  // namespace

int main(int argc, char** argv) {
    Case c{640, 2560, 16};
    int iters = 300;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) c.rows = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) c.k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--groups") && i + 1 < argc) c.groups = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: %s [--rows N] [--k N] [--groups G] [--iters N]\n", argv[0]); return 2; }
    }
    const int n_blk = c.k / QK;
    const size_t n_rows = (size_t) c.rows * c.groups;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> n4(-8, 7);
    std::normal_distribution<float> nf(0.0f, 1.0f);

    // ---- weights: one value array, two encodings (offset binary for dp4a, two's complement for dot8)
    std::vector<int8_t> wv(n_rows * c.k);
    for (auto& v : wv) v = (int8_t) n4(rng);
    auto pack_weights = [&](bool twos) {
        std::vector<BlockW> hw(n_rows * n_blk);
        for (size_t r = 0; r < n_rows; ++r)
            for (int b = 0; b < n_blk; ++b) {
                BlockW& o = hw[r * n_blk + b];
                o.d = 0.02f;
                for (int i = 0; i < QK / 8; ++i) {
                    uint32_t v = 0;
                    for (int j = 0; j < 4; ++j) {
                        const int lo = wv[r * c.k + b * QK + i * 8 + j];
                        const int hi = wv[r * c.k + b * QK + i * 8 + 4 + j];
                        v |= (uint32_t) ((twos ? (lo & 0xF) : (lo + 8)) & 0xF) << (8 * j);
                        v |= (uint32_t) ((twos ? (hi & 0xF) : (hi + 8)) & 0xF) << (8 * j + 4);
                    }
                    o.q[i] = v;
                }
            }
        return hw;
    };
    const std::vector<BlockW> hw_dp4a = pack_weights(false);
    const std::vector<BlockW> hw_dot8 = pack_weights(true);

    // ---- activations: one float array, quantised to 8 bits (dp4a) and to 4 bits (dot8)
    std::vector<std::vector<float>> xf(n_blk, std::vector<float>(QK));
    std::vector<BlockQ8> hx8(n_blk);
    std::vector<BlockQ4> hx4(n_blk);
    for (int b = 0; b < n_blk; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < QK; ++i) { xf[b][i] = nf(rng); amax = std::max(amax, std::fabs(xf[b][i])); }
        hx8[b].d = amax / 127.0f;
        for (int i = 0; i < QK; ++i) hx8[b].qs[i] = (int8_t) std::lround(xf[b][i] / hx8[b].d);
        hx4[b].d = amax / 7.0f;
        for (int i = 0; i < QK / 8; ++i) {
            uint32_t v = 0;
            for (int j = 0; j < 4; ++j) {
                const int lo = std::max(-8, std::min(7, (int) std::lround(xf[b][i * 8 + j] / hx4[b].d)));
                const int hi = std::max(-8, std::min(7, (int) std::lround(xf[b][i * 8 + 4 + j] / hx4[b].d)));
                v |= (uint32_t) (lo & 0xF) << (8 * j);
                v |= (uint32_t) (hi & 0xF) << (8 * j + 4);
            }
            hx4[b].q[i] = v;
        }
    }

    // ---- FP32 reference: the same weight values and the same float activations
    std::vector<float> ref(n_rows, 0.0f);
    for (size_t r = 0; r < n_rows; ++r) {
        double s = 0.0;
        for (int b = 0; b < n_blk; ++b) {
            double isum = 0.0;
            for (int i = 0; i < QK; ++i) isum += (double) wv[r * c.k + b * QK + i] * (double) xf[b][i];
            s += 0.02 * isum;
        }
        ref[r] = (float) s;
    }

    void* dw1 = nullptr; void* dw2 = nullptr; void* dx8 = nullptr; void* dx4 = nullptr; float* dy = nullptr;
    HIP_CHECK(hipMalloc(&dw1, hw_dp4a.size() * sizeof(BlockW)));
    HIP_CHECK(hipMalloc(&dw2, hw_dot8.size() * sizeof(BlockW)));
    HIP_CHECK(hipMalloc(&dx8, hx8.size() * sizeof(BlockQ8)));
    HIP_CHECK(hipMalloc(&dx4, hx4.size() * sizeof(BlockQ4)));
    HIP_CHECK(hipMalloc(&dy, n_rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(dw1, hw_dp4a.data(), hw_dp4a.size() * sizeof(BlockW), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dw2, hw_dot8.data(), hw_dot8.size() * sizeof(BlockW), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx8, hx8.data(), hx8.size() * sizeof(BlockQ8), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx4, hx4.data(), hx4.size() * sizeof(BlockQ4), hipMemcpyHostToDevice));

    std::vector<float> out(n_rows);
    auto error = [&](const char* label) {
        HIP_CHECK(hipMemcpy(out.data(), dy, n_rows * sizeof(float), hipMemcpyDeviceToHost));
        double num = 0.0, den = 0.0;
        for (size_t r = 0; r < n_rows; ++r) { num += (double) (out[r] - ref[r]) * (out[r] - ref[r]); den += (double) ref[r] * ref[r]; }
        return std::sqrt(num / std::max(den, 1e-30));
    };

    const double t4 = bench(false, c, dw1, dx8, dy, iters);
    const double e4 = error("dp4a");
    const double t8 = bench(true, c, dw2, dx4, dy, iters);
    const double e8 = error("dot8");

    const double macs = (double) n_rows * (double) c.k;
    std::printf("rows %d x groups %d, K %d  (%zu row-dots, %.1f M weights, 4-bit = %.1f MB per pass)\n", c.rows, c.groups,
                c.k, n_rows, macs / 1e6, macs / 2.0 / 1e6);
    std::printf("  dp4a : %8.4f ms   %7.1f G MAC/s   %4.0f dots/row   error vs FP32 %.3f %%\n", t4, macs / (t4 * 1e6),
                (double) c.k / 4.0, e4 * 100.0);
    std::printf("  dot8 : %8.4f ms   %7.1f G MAC/s   %4.0f dots/row   error vs FP32 %.3f %%\n", t8, macs / (t8 * 1e6),
                (double) c.k / 8.0, e8 * 100.0);
    std::printf("  dot8 vs dp4a: %+.1f %% throughput\n", (t4 / t8 - 1.0) * 100.0);
    return 0;
}
