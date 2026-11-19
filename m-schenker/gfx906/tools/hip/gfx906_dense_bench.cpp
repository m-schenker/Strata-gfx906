// tools/hip/gfx906_dense_bench.cpp - the dense paths' kernel arm: native_mmvq (quantized GEMV/MMVQ) and
// native_router_top10 (the top-10 router), timed through the engine's own launchers out of libstrata_kernels.a.
//
//     gfx906_dense_bench [--type 14] [--k 2560] [--n 5120] [--cols 1] [--iters 200] [--what mmvq|router|both]
//
// Why: the trellis expert kernels have native_expert_bench, but the dense projections and the router - the two paths
// every layer hits once per token - had no arm at all, so "the router launch shape is a stall" and "software-pipeline the
// residual GEMV" could not be measured on this card.  This links the same launchers the engine calls, fills the weight
// buffer with a byte pattern (the shapes and instruction mix are what is being timed, not the values), and reports
// microseconds per launch and microseconds per call site at the model's real sizes.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <functional>
#include <vector>

#include <strata/kernels/native_mmvq.hpp>
#include <strata/kernels/native_router.hpp>

#define HIP_CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP %s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

using namespace strata::kernels;

namespace {

double time_launches(int iters, hipStream_t st, const std::function<void()>& f) {
    HIP_CHECK(hipDeviceSynchronize());
    f();
    HIP_CHECK(hipStreamSynchronize(st));
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    HIP_CHECK(hipEventRecord(a, st));
    for (int i = 0; i < iters; ++i) f();
    HIP_CHECK(hipEventRecord(b, st));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return (double) ms * 1000.0 / iters;   // us per launch
}

const char* type_name(int t) {
    switch (t) {
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 8: return "Q8_0";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 20: return "IQ4_NL";
        case 23: return "IQ4_XS";
        case 42: return "Q2_0";
        default: return "?";
    }
}

}  // namespace

int main(int argc, char** argv) {
    int type = 14, n_in = 2560, n_out = 5120, cols = 1, iters = 200;
    std::string what = "both";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--type") && i + 1 < argc) type = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) n_in = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n_out = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--cols") && i + 1 < argc) cols = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--what") && i + 1 < argc) what = argv[++i];
        else { std::fprintf(stderr, "usage: %s [--type T] [--k N] [--n N] [--cols C] [--iters N] [--what mmvq|router|both]\n",
                            argv[0]); return 2; }
    }

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    if (what == "mmvq" || what == "both") {
        if (!native_mmvq_supported(type)) {
            std::printf("type %d (%s) is not a native_mmvq type on this build\n", type, type_name(type));
        } else {
            const size_t wb = native_mmvq_weight_bytes(type, n_in, n_out);
            const size_t xb = native_q8_1_bytes(n_in, cols);
            std::vector<uint8_t> hw(wb);
            for (size_t i = 0; i < wb; ++i) hw[i] = (uint8_t) (0x11 + 0x07 * (i % 13));   // any bytes: the shape is timed
            std::vector<float> hx((size_t) n_in * cols);
            for (size_t i = 0; i < hx.size(); ++i) hx[i] = 0.01f * (float) ((int) (i * 37 % 101) - 50);

            void* dw = nullptr;
            void* dx = nullptr;
            void* dq = nullptr;
            float* dy = nullptr;
            HIP_CHECK(hipMalloc(&dw, wb));
            HIP_CHECK(hipMalloc(&dx, hx.size() * sizeof(float)));
            HIP_CHECK(hipMalloc(&dq, xb));
            HIP_CHECK(hipMalloc(&dy, (size_t) n_out * cols * sizeof(float)));
            HIP_CHECK(hipMemcpy(dw, hw.data(), wb, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dx, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice));
            native_quantize_q8_1((const float*) dx, dq, n_in, cols, (void*) stream);
            HIP_CHECK(hipDeviceSynchronize());

            const double us = time_launches(iters, stream, [&] {
                native_mmvq(type, dw, dq, dy, n_in, n_out, cols, (void*) stream);
            });
            const double macs = (double) n_in * n_out * cols;
            std::printf("mmvq   %-6s k %d n %d cols %d : %8.2f us/launch  %7.1f G MAC/s  %.1f GB/s of weights\n",
                        type_name(type), n_in, n_out, cols, us, macs / (us * 1e3), (double) wb / (us * 1e3));
            HIP_CHECK(hipFree(dw));
            HIP_CHECK(hipFree(dx));
            HIP_CHECK(hipFree(dq));
            HIP_CHECK(hipFree(dy));
        }
    }

    if (what == "router" || what == "both") {
        float* dl = nullptr;
        int32_t* di = nullptr;
        float* dwt = nullptr;
        HIP_CHECK(hipMalloc(&dl, 512 * 4));
        HIP_CHECK(hipMalloc(&di, 10 * 4));
        HIP_CHECK(hipMalloc(&dwt, 10 * 4));
        std::vector<float> hl(512);
        for (int i = 0; i < 512; ++i) hl[i] = 0.001f * (float) ((i * 71) % 97);
        HIP_CHECK(hipMemcpy(dl, hl.data(), 512 * 4, hipMemcpyHostToDevice));
        const double us1 = time_launches(iters * 10, stream, [&] { native_router_top10(dl, di, dwt, (void*) stream); });
        std::printf("router top10 over 512 experts, 1 token  : %8.3f us/launch\n", us1);
        // 48 layers x 1 call for attention + 1 for the MLP: the per-token cost of this kernel in the model
        std::printf("       -> %6.3f ms per token at 96 calls (48 layers x 2)\n", us1 * 96.0 / 1000.0);
        HIP_CHECK(hipFree(dl));
        HIP_CHECK(hipFree(di));
        HIP_CHECK(hipFree(dwt));
    }
    return 0;
}
