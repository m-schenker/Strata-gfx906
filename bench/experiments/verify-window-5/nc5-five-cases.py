p=sys.argv[1] if len(sys.argv) > 1 else 'src/kernels/cuda/iq_kernels.cu'
import sys
s=open(p).read()
def sub1(old,new,label):
    global s
    assert s.count(old)==1, (label, s.count(old))
    s=s.replace(old,new); print("patched:",label)

sub1("constexpr int GRP_NC = 4;",
     "constexpr int GRP_NC = 4;\n// local experiment (not in the PR): the 5-entry verify window in ONE pass, in the three branches\n// an IQ3_S/IQ4_NL pack (n_embd 2560, n_ff 640) actually runs: gate/up staged (xb = 80), gate/up global, down staged.\nconstexpr int NC5 = 5;",
     "NC5")

# gate/up STAGED branch (xb <= 80, the one our model runs): buffer, stride, n==5 case
sub1("            __shared__ uint32_t s_x[GRP_NC * 80 * 9];",
     "            __shared__ uint32_t s_x[NC5 * 80 * 9];", "gu staged buffer")
sub1("""                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    __syncthreads();""",
     """                for (int e = e0; e < e1; e += NC5) {
                    const int n = min(NC5, e1 - e);
                    __syncthreads();""", "gu staged stride")
sub1("""                        } else {
                            const int off4[4] = { 0, xb, 2 * xb, 3 * xb };""",
     """                        } else if (n == 5) {
                            const int off5[5] = { 0, xb, 2 * xb, 3 * xb, 4 * xb };
                            float s5[5];
                            row_dot_multi<TG, 5, true, STAGE_GRID>(wr, xs, off5, 5, nb, lane, s5, s_grid);
                            if (lane < 5) dst[(size_t) (e + lane) * L.n_ff + r] = s5[lane] * sc;
                        } else {
                            const int off4[4] = { 0, xb, 2 * xb, 3 * xb };""", "gu staged n==5")

# gate/up global branch (kept for other geometries)
sub1("""        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            if (n == 1) {
                const int off1[1] = { ent_tok[e] * xb };""",
     """        for (int e = e0; e < e1; e += NC5) {
            const int n = min(NC5, e1 - e);
            if (n == 1) {
                const int off1[1] = { ent_tok[e] * xb };""", "gu global stride")
sub1("""            } else {
                const int off4[4] = { ent_tok[e] * xb, ent_tok[e + 1] * xb, ent_tok[e + 2] * xb, ent_tok[e + 3] * xb };""",
     """            } else if (n == 5) {
                const int off5[5] = { ent_tok[e] * xb, ent_tok[e + 1] * xb, ent_tok[e + 2] * xb, ent_tok[e + 3] * xb,
                                      ent_tok[e + 4] * xb };
                float s5[5];
                row_dot_multi<TG, 5, true, STAGE_GRID>(wr, xq, off5, 5, nb, lane, s5, s_grid);
                if (lane < 5) dst[(size_t) (e + lane) * L.n_ff + r] = s5[lane] * sc;
            } else {
                const int off4[4] = { ent_tok[e] * xb, ent_tok[e + 1] * xb, ent_tok[e + 2] * xb, ent_tok[e + 3] * xb };""",
     "gu global n==5")

# down staged branch (hb == 20)
sub1("    __shared__ uint32_t s_hq_buf[GRP_NC * 20 * 9];",
     "    __shared__ uint32_t s_hq_buf[NC5 * 20 * 9];", "down buffer")
sub1("""            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = min(GRP_NC, e1 - e);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);""",
     """            for (int e = e0; e < e1; e += NC5) {
                const int n = min(NC5, e1 - e);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);""", "down stride")
sub1("""                    } else {
                        int off[GRP_NC];
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * 20;
                        float s[GRP_NC];
                        row_dot_multi<TD, GRP_NC, false, STAGE_TQ>(wr, s_hq, off, n, nb, lane, s, s_grid);
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c)
                            if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c] * sd;""",
     """                    } else {
                        int off[NC5];
#pragma unroll
                        for (int c = 0; c < NC5; ++c) off[c] = min(c, n - 1) * 20;
                        float s[NC5];
                        row_dot_multi<TD, NC5, false, STAGE_TQ>(wr, s_hq, off, n, nb, lane, s, s_grid);
#pragma unroll
                        for (int c = 0; c < NC5; ++c)
                            if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c] * sd;""",
     "down arrays")
open(p,'w').write(s); print("ok")
