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
                          int n_row, int n_blk, const int* __restrict__ sx) {
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
            // The nibbles are unsigned 0..15 here and the -8 bias is taken out once per block, below: that is how a
            // real Q4_0 kernel does it, and it costs two masks per 8 values instead of a per-byte sign-extension (a
            // per-byte `nibble - 8` BORROWS across bytes: 0x03030303 - 0x08080808 = 0xFAFAFAFB, three of the four
            // values wrong - which is what this tool measured before, as a 17-19 % "error" on the dp4a arm).
            const uint32_t lo = v & 0x0F0F0F0Fu;
            const uint32_t hi = (v >> 4) & 0x0F0F0F0Fu;
            s = __builtin_amdgcn_sdot4((int) lo, x8[2 * i], s, false);
            s = __builtin_amdgcn_sdot4((int) hi, x8[2 * i + 1], s, false);
        }
        acc += xb->d * wb->d * (float) (s - 8 * sx[b]);   // the Q4_0 bias: (sum n*a) - 8*(sum a)
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

void* g_sx = nullptr;   // the per-block activation sums the dp4a arm's bias correction needs

double bench(bool dot8, const Case& c, void* w, void* x, float* y, int iters) {
    auto launch = [&](int reps) {
        const int wpb = 4;
        const unsigned grid = (unsigned) ((c.rows * c.groups + wpb - 1) / wpb);
        for (int i = 0; i < reps; ++i) {
            if (dot8)
                gemv_dot8<<<grid, 64 * wpb>>>((const BlockW*) w, (const BlockQ4*) x, y, c.rows * c.groups, c.k / QK);
            else
                gemv_dp4a<<<grid, 64 * wpb>>>((const BlockW*) w, (const BlockQ8*) x, y, c.rows * c.groups, c.k / QK,
                                               (const int*) g_sx);
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


// ---------------------------------------------------------------- real weights out of a GGUF
// The synthetic matrix above answers "what does the instruction cost"; a real Q4_0 tensor answers "what are the rows
// actually like" (per-block FP16 scales that vary by 100x, codebook-free nibbles).  Only what a Q4_0 tensor needs is
// parsed: the metadata block is skipped by its own type table, then the tensor list gives name/dims/type/offset.
struct GgufTensor { std::string name; uint32_t type = 0; uint64_t dim[4] = {0, 0, 0, 0}; uint32_t n_dims = 0; uint64_t off = 0; };

struct Gguf {
    uint64_t data_at = 0;
    std::vector<GgufTensor> tensors;

    static uint64_t rd(FILE* f, void* dst, uint64_t n) { return std::fread(dst, 1, n, f); }
    static std::string rs(FILE* f) { uint64_t n = 0; if (rd(f, &n, 8) != 8) return {}; std::string s(n, '\0'); if (n) rd(f, &s[0], n); return s; }

    static void skip_value(FILE* f, uint32_t type) {
        static const int sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (type == 8) { const std::string v = rs(f); (void) v; return; }
        if (type == 9) { uint32_t t = 0; uint64_t n = 0; rd(f, &t, 4); rd(f, &n, 8); for (uint64_t i = 0; i < n; ++i) skip_value(f, t); return; }
        if (type > 12) return;
        char buf[8]; rd(f, buf, (uint64_t) sz[type]);
    }

    bool open(const char* path, std::string& err) {
        FILE* f = std::fopen(path, "rb");
        if (!f) { err = "cannot open"; return false; }
        char magic[4]; uint32_t ver = 0; uint64_t nt = 0, nkv = 0;
        rd(f, magic, 4); rd(f, &ver, 4); rd(f, &nt, 8); rd(f, &nkv, 8);
        if (std::memcmp(magic, "GGUF", 4) != 0) { err = "not a GGUF"; std::fclose(f); return false; }
        uint64_t align = 32;
        for (uint64_t i = 0; i < nkv; ++i) {
            const std::string key = rs(f);
            uint32_t type = 0; rd(f, &type, 4);
            if (key == "general.alignment" && type == 4) { uint32_t a = 0; rd(f, &a, 4); align = a; }
            else skip_value(f, type);
        }
        for (uint64_t i = 0; i < nt; ++i) {
            GgufTensor t; t.name = rs(f); rd(f, &t.n_dims, 4);
            for (uint32_t d = 0; d < t.n_dims && d < 4; ++d) rd(f, &t.dim[d], 8);
            rd(f, &t.type, 4); rd(f, &t.off, 8);
            tensors.push_back(t);
        }
        data_at = (uint64_t) std::ftell(f);
        data_at = (data_at + align - 1) / align * align;
        std::fclose(f);
        return true;
    }
};

constexpr uint32_t kGgufQ4_0 = 2;   // ggml type 2: 32 values, one f16 scale, 16 bytes of offset-binary nibbles

// the largest 2-D Q4_0 tensor of the file, and its [rows, k]
bool find_q4_0(const char* path, GgufTensor& out, uint64_t& rows, uint64_t& k, std::string& err) {
    Gguf g;
    if (!g.open(path, err)) return false;
    bool found = false;
    for (const auto& t : g.tensors) {
        if (t.type != kGgufQ4_0 || t.n_dims != 2 || t.dim[0] % QK != 0 || t.dim[1] < 8) continue;
        if (!found || t.dim[0] * t.dim[1] > rows * k) { out = t; rows = t.dim[1]; k = t.dim[0]; found = true; }
    }
    if (!found) { err = "no 2-D Q4_0 tensor"; return false; }
    out.off += g.data_at;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Case c{640, 2560, 16};
    int iters = 300;
    const char* gguf_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) c.rows = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) c.k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--groups") && i + 1 < argc) c.groups = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--gguf") && i + 1 < argc) gguf_path = argv[++i];
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
    std::vector<float> wd((size_t) n_rows * n_blk, 0.02f);   // the per-block FP16 scale of a Q4_0 row
    if (gguf_path) {
        // real Q4_0 rows: keep the file's own nibbles and FP16 scales, stream c.groups of them at a time
        GgufTensor t; uint64_t rows = 0, kk = 0; std::string err;
        if (!find_q4_0(gguf_path, t, rows, kk, err)) { std::fprintf(stderr, "%s: %s\n", gguf_path, err.c_str()); return 2; }
        FILE* f = std::fopen(gguf_path, "rb");
        if (!f) { std::fprintf(stderr, "cannot reopen %s\n", gguf_path); return 2; }
        const uint64_t want_rows = std::min<uint64_t>(n_rows, rows), reuse = (n_rows + rows - 1) / rows;
        for (uint64_t r = 0; r < want_rows; ++r) {
            std::vector<uint8_t> blk((size_t) n_blk * 18);
            std::fseek(f, (long) (t.off + ((r % rows) * (kk / QK) * 18)), SEEK_SET);
            if (std::fread(blk.data(), 18, (size_t) n_blk, f) != (size_t) n_blk) { std::fprintf(stderr, "short read\n"); return 2; }
            for (int64_t b = 0; b < n_blk; ++b) {
                const uint8_t* p = blk.data() + b * 18;
                const _Float16 d16 = *reinterpret_cast<const _Float16*>(p);
                wd[(size_t) r * n_blk + b] = (float) d16;
                for (int i = 0; i < QK / 8; ++i) {   // the Q4_0 packing: byte j low = value 8i+j, high = 8i+4+j
                    const uint32_t v = uint32_t(p[2 + i * 4]) | (uint32_t(p[3 + i * 4]) << 8) |
                                       (uint32_t(p[4 + i * 4]) << 16) | (uint32_t(p[5 + i * 4]) << 24);
                    for (int j = 0; j < 4; ++j) {
                        wv[(size_t) r * c.k + b * QK + i * 8 + j] = (int8_t) (((v >> (8 * j)) & 0xF) - 8);
                        wv[(size_t) r * c.k + b * QK + i * 8 + 4 + j] = (int8_t) (((v >> (8 * j + 4)) & 0xF) - 8);
                    }
                }
            }
        }
        // a group index above the file's row count reuses rows (the arms only need the same bytes, not distinct rows)
        for (uint64_t r = want_rows; r < n_rows; ++r)
            for (int b = 0; b < n_blk; ++b) {
                wd[(size_t) r * n_blk + b] = wd[(size_t) (r % rows == 0 ? r - 1 : r % rows) * n_blk + b];
                for (int i = 0; i < QK; ++i)
                    wv[(size_t) r * c.k + b * QK + i] = wv[(size_t) (r % rows) * c.k + b * QK + i];
            }
        std::fclose(f);
        std::printf("weights: %s `%s` [%llu x %llu], using %llu rows%s\n", gguf_path, t.name.c_str(),
                    (unsigned long long) rows, (unsigned long long) kk, (unsigned long long) want_rows,
                    reuse > 1 ? " (cycled)" : "");
    }
    auto pack_weights = [&](bool twos) {
        std::vector<BlockW> hw(n_rows * n_blk);
        for (size_t r = 0; r < n_rows; ++r)
            for (int b = 0; b < n_blk; ++b) {
                BlockW& o = hw[r * n_blk + b];
                o.d = wd[r * n_blk + b];
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
            s += (double) wd[r * n_blk + b] * isum;
        }
        ref[r] = (float) s;
    }

    std::vector<int> hsx(n_blk, 0);
    for (int b = 0; b < n_blk; ++b) for (int i = 0; i < QK; ++i) hsx[b] += hx8[b].qs[i];

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
    HIP_CHECK(hipMalloc(&g_sx, hsx.size() * sizeof(int)));
    HIP_CHECK(hipMemcpy(g_sx, hsx.data(), hsx.size() * sizeof(int), hipMemcpyHostToDevice));

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
    if (std::getenv("DOT8_DEBUG")) {   // row 0..3 of each arm against the FP32 reference
        HIP_CHECK(hipMemcpy(out.data(), dy, n_rows * sizeof(float), hipMemcpyDeviceToHost));
        for (size_t r = 0; r < std::min<size_t>(4, n_rows); ++r) std::printf("  dot8  row %zu: %.6f   ref %.6f\n", r, out[r], ref[r]);
    }
    std::printf("rows %d x groups %d, K %d  (%zu row-dots, %.1f M weights, 4-bit = %.1f MB per pass)\n", c.rows, c.groups,
                c.k, n_rows, macs / 1e6, macs / 2.0 / 1e6);
    std::printf("  dp4a : %8.4f ms   %7.1f G MAC/s   %4.0f dots/row   error vs FP32 %.3f %%\n", t4, macs / (t4 * 1e6),
                (double) c.k / 4.0, e4 * 100.0);
    std::printf("  dot8 : %8.4f ms   %7.1f G MAC/s   %4.0f dots/row   error vs FP32 %.3f %%\n", t8, macs / (t8 * 1e6),
                (double) c.k / 8.0, e8 * 100.0);
    std::printf("  dot8 vs dp4a: %+.1f %% throughput\n", (t4 / t8 - 1.0) * 100.0);
    return 0;
}
