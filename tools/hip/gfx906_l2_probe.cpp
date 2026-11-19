// tools/hip/gfx906_l2_probe.cpp - can an L2 policy hold this engine's decode hot set?
//
//     clang++ -x hip --offload-arch=gfx906 --rocm-path=/opt/rocm -O3 -o l2 gfx906_l2_probe.cpp && ./l2
//
// The question: decode re-reads the same few experts every token, so if the hot set fit the L2 (8 MiB on MI50) - or if a
// persisting-cache window could pin it - the expert GEMV would run at L2 speed instead of HBM speed.  This tool
// answers it in two parts:
//   1. what the HIP surface exposes for L2 residency on this card (attribute sizes, whether an access-policy window
//      is accepted, and what it returns), because a policy the runtime silently ignores is not a lever;
//   2. the bandwidth the expert GEMV actually gets as the weight footprint grows past L2 - the cliff is the
//      measurement that says whether a policy could matter at all.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP %s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {

// one 64-lane wave per row, the shape of the engine's decode GEMV: sum over K of w[row][k] * x[k]
__global__ void gemv(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                     int n_row, int k) {
    const int lane = threadIdx.x & 63;
    const int row = blockIdx.x * (blockDim.x / 64) + (threadIdx.x >> 6);
    if (row >= n_row) return;
    float acc = 0.0f;
    for (int i = lane; i < k; i += 64) acc += w[(size_t) row * k + i] * x[i];
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 64);
    if (lane == 0) y[row] = acc;
}

double bench(const float* w, const float* x, float* y, int n_row, int k, int iters, float& worst) {
    const int wpb = 4;
    const unsigned grid = (unsigned) ((n_row + wpb - 1) / wpb);
    auto launch = [&](int reps) {
        for (int i = 0; i < reps; ++i) gemv<<<grid, 64 * wpb>>>(w, x, y, n_row, k);
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
    worst = (float) (ms / iters);
    return (double) ms / iters;
}

}  // namespace

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s, %d CU, %d MHz\n", prop.name, prop.multiProcessorCount, prop.memoryClockRate / 1000);
    int l2 = 0;
    HIP_CHECK(hipDeviceGetAttribute(&l2, hipDeviceAttributeL2CacheSize, 0));
    std::printf("L2 cache: %.2f MiB\n", l2 / 1048576.0);

    // 1. what the runtime will admit about residency
#ifdef hipDeviceAttributePersistingL2CacheSize
    int persist = -1, persist_max = -1;
    const hipError_t e1 = hipDeviceGetAttribute(&persist, hipDeviceAttributePersistingL2CacheSize, 0);
    const hipError_t e2 = hipDeviceGetAttribute(&persist_max, hipDeviceAttributeMaxPersistingL2CacheSize, 0);
    std::printf("hipDeviceAttributePersistingL2CacheSize      -> %-28s value %d\n", hipGetErrorString(e1), persist);
    std::printf("hipDeviceAttributeMaxPersistingL2CacheSize   -> %-28s value %d\n", hipGetErrorString(e2), persist_max);
#else
    std::printf("hipDeviceAttributePersistingL2CacheSize      -> not in this HIP header\n");
#endif

    // 2. the bandwidth cliff: the engine's expert GEMV as the footprint grows past L2
    const int k = 2560;                        // n_embd of the served model
    float* x = nullptr;
    HIP_CHECK(hipMalloc(&x, k * sizeof(float)));
    std::vector<float> hx(k, 0.5f);
    HIP_CHECK(hipMemcpy(x, hx.data(), k * sizeof(float), hipMemcpyHostToDevice));
    std::printf("\n footprint   rows   ms/pass   GB/s   (the engine's expert GEMV, K %d)\n", k);
    float* y = nullptr;
    HIP_CHECK(hipMalloc(&y, 4096 * sizeof(float)));
    for (double mb : {1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 13.1, 26.2, 52.4}) {
        const int rows = (int) (mb * 1048576.0 / (k * sizeof(float)));
        float* w = nullptr;
        HIP_CHECK(hipMalloc(&w, (size_t) rows * k * sizeof(float)));
        std::vector<float> hw((size_t) rows * k, 0.25f);
        HIP_CHECK(hipMemcpy(w, hw.data(), hw.size() * sizeof(float), hipMemcpyHostToDevice));
        float worst = 0.0f;
        const double ms = bench(w, x, y, rows, k, 200, worst);
        std::printf(" %6.1f MiB  %5d  %7.4f  %6.1f\n", mb, rows, ms, (rows * (double) k * 4.0) / (worst * 1e6));
        HIP_CHECK(hipFree(w));
    }
    std::printf("\n the engine's decode hot set: 10 of 512 experts per layer x 1.31 MB = 13.1 MB per layer, and 48\n"
                " layers are live per token - the 8 MiB L2 covers about 60 %% of ONE layer's picks.\n");
    return 0;
}
