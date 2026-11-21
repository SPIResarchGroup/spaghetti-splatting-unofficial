// Spaghetti Splatting: tile-based rasterizer for kernel-convolved line segments.
//
// Every primitive is a 3D segment [a, b] convolved with a kernel. After the
// per-segment EWA linearisation at the midpoint (done in PyTorch), the optical
// depth of a pixel p is
//
//     tau(p) = rho * I(P, R, K),   I = \int_0^1 k( q(lambda) ) dlambda,
//     q(lambda) = || g0 + lambda (g1 - g0) ||_Q^2 = P + 2 R lambda + K lambda^2
//
// with g0 = (p - pa) * ca, g1 = (p - pb) * cb, ca = z_a / z_m, cb = z_b / z_m,
// and Q = 0.5 * inverse(2D covariance at the midpoint). The inverse-depth
// scaling makes this perspective-correct along the segment (lambda is the 3D
// segment parameter). For the Gaussian kernel k(q) = exp(-q) the integral has a
// closed form via erf (paper Eq. 19); for the Beta kernel k(q) = (1 - 2q/9)_+^beta
// we use 8-point Gauss-Legendre quadrature on the exact support interval.
//
// alpha = min(0.99, 1 - exp(-tau)), front-to-back compositing as in 3DGS.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define TILE 16
#define BLOCK_SIZE (TILE * TILE)
#define NCH 3
#define NGRAD 14  // pa(2) pb(2) ca cb Q(3) rho color(3) beta

enum KernelType { GAUSSIAN = 0, BETA = 1 };

__constant__ float GL_X[8] = {-0.9602898564975363f, -0.7966664774136267f, -0.5255324099163290f,
                              -0.1834346424956498f, 0.1834346424956498f,  0.5255324099163290f,
                              0.7966664774136267f,  0.9602898564975363f};
__constant__ float GL_W[8] = {0.1012285362903763f, 0.2223810344533745f, 0.3137066458778873f,
                              0.3626837833783620f, 0.3626837833783620f, 0.3137066458778873f,
                              0.2223810344533745f, 0.1012285362903763f};

#define SQRT_PI 1.7724538509055159f
#define GAUSS_KEPS 1e-4f
#define BETA_QMAX 4.5f  // support of (1 - 2q/9)_+ : q < 4.5  (Mahalanobis radius 3)

// ---------------------------------------------------------------------------
// 1D integrals along the segment
// ---------------------------------------------------------------------------

// erf(A) + erf(B) with A + B = kappa > 0, evaluated without cancellation.
__device__ __forceinline__ float erf_sum(float A, float B) {
    if (A >= 0.f && B >= 0.f) return erff(A) + erff(B);
    if (A < 0.f) return erfcf(-A) - erfcf(B);
    return erfcf(-B) - erfcf(A);
}

// I = \int_0^1 exp(-(P + 2R l + K l^2)) dl and its partials.
// P1 = P + 2R + K (= q(1)) and E = P - R^2/K (= min_l q(l), perpendicular distance) are passed in, evaluated
// from vectors by the caller to avoid float32 cancellation for long, thin segments.
template <bool GRAD>
__device__ __forceinline__ void gauss_integral(float P, float R, float K, float P1, float E, float& I, float& dP,
                                               float& dR, float& dK) {
    const float e0 = expf(-P);
    if (K < GAUSS_KEPS) {
        // second-order expansion around a point primitive (kappa -> 0)
        I = e0 * (1.f - R - K * (1.f / 3.f) + R * R * (2.f / 3.f));
        if (GRAD) {
            dP = -I;
            dR = e0 * (-1.f + R * (4.f / 3.f));
            dK = -e0 * (1.f / 3.f);
        }
        return;
    }
    const float kappa = sqrtf(K);
    const float A = -R / kappa;
    const float B = kappa - A;
    const float Z = erf_sum(A, B);
    I = expf(-E) * (0.5f * SQRT_PI / kappa) * Z;
    if (GRAD) {
        const float e1 = expf(-P1);  // integrand at l = 1
        const float iK = 1.f / K;
        dP = -I;
        dR = 2.f * R * iK * I + (e1 - e0) * iK;
        dK = -(R * R * iK * iK + 0.5f * iK) * I + 0.5f * R * (e0 - e1) * iK * iK + 0.5f * e1 * iK;
    }
}

#define BETA_KSTABLE 1.f  // for K >= this, use q(l) = E + K (l - l*)^2 (E evaluated from vectors)

// Support interval {l in [0,1] : q(l) < 4.5}. Treated as a constant w.r.t. the
// parameters (Leibniz: the integrand vanishes at the support boundary).
__device__ __forceinline__ bool beta_support(float P, float R, float K, float E, float& lo, float& hi) {
    if (K >= BETA_KSTABLE) {  // long segment: avoid the cancellation in R^2 - K (P - 4.5)
        if (E >= BETA_QMAX) return false;
        const float ls = -R / K, h = sqrtf((BETA_QMAX - E) / K);
        lo = fmaxf(0.f, ls - h);
        hi = fminf(1.f, ls + h);
        return hi > lo;
    }
    const float c0 = P - BETA_QMAX;
    if (K < 1e-12f && fabsf(R) < 1e-12f) {
        lo = 0.f;
        hi = 1.f;
        return c0 < 0.f;
    }
    const float disc = R * R - K * c0;
    if (disc <= 0.f) return false;
    const float sq = sqrtf(disc);
    const float qq = -(R + copysignf(sq, R));
    const float r1 = (K > 1e-12f) ? qq / K : (qq > 0.f ? 1e30f : -1e30f);
    const float r2 = (fabsf(qq) > 1e-30f) ? c0 / qq : (c0 < 0.f ? -1e30f : 1e30f);
    lo = fmaxf(0.f, fminf(r1, r2));
    hi = fminf(1.f, fmaxf(r1, r2));
    return hi > lo;
}

template <bool GRAD>
__device__ __forceinline__ void beta_integral(float P, float R, float K, float E, float beta, float& I, float& dP,
                                              float& dR, float& dK, float& db) {
    I = 0.f;
    if (GRAD) dP = dR = dK = db = 0.f;
    float lo, hi;
    if (!beta_support(P, R, K, E, lo, hi)) return;
    const float half = 0.5f * (hi - lo), mid = 0.5f * (hi + lo);
    const bool stable = K >= BETA_KSTABLE;
    const float ls = stable ? -R / K : 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float l = mid + half * GL_X[i];
        const float q = stable ? E + K * (l - ls) * (l - ls) : P + l * (2.f * R + l * K);
        const float base = 1.f - (2.f / 9.f) * q;
        if (base <= 0.f) continue;
        const float lb = logf(base);
        const float f = expf(beta * lb);
        const float w = half * GL_W[i];
        I += w * f;
        if (GRAD) {
            const float dfdq = -(2.f / 9.f) * beta * f / base;
            dP += w * dfdq;
            dR += w * dfdq * 2.f * l;
            dK += w * dfdq * l * l;
            db += w * f * lb;
        }
    }
}

// ---------------------------------------------------------------------------
// per primitive x pixel evaluation
// ---------------------------------------------------------------------------

struct Prim {
    float pax, pay, pbx, pby, ca, cb, qa, qb, qc, rho, beta;
    float col[NCH];
};

template <int KT, bool GRAD>
__device__ __forceinline__ float eval_tau(const Prim& g, float px, float py, float& I, float& dP, float& dR,
                                          float& dK, float& db, float& g0x, float& g0y, float& dlx,
                                          float& dly) {
    const float rax = px - g.pax, ray = py - g.pay;
    const float rbx = px - g.pbx, rby = py - g.pby;
    g0x = rax * g.ca;
    g0y = ray * g.ca;
    const float g1x = rbx * g.cb, g1y = rby * g.cb;
    dlx = g1x - g0x;
    dly = g1y - g0y;
    const float Qg0x = g.qa * g0x + g.qb * g0y, Qg0y = g.qb * g0x + g.qc * g0y;
    const float Qdlx = g.qa * dlx + g.qb * dly, Qdly = g.qb * dlx + g.qc * dly;
    const float P = g0x * Qg0x + g0y * Qg0y;
    const float R = dlx * Qg0x + dly * Qg0y;
    const float K = fmaxf(dlx * Qdlx + dly * Qdly, 0.f);
    if (KT == GAUSSIAN && !GRAD && P > 60.f && K < 1.f) {  // cheap early-out far from short segments
        I = 0.f;
        return 0.f;
    }
    // E = min_l q(l) (squared Q-distance to the segment's line), from vectors to avoid cancellation
    float E = 0.f;
    if (K >= GAUSS_KEPS) {
        const float ub = -R / K;
        const float ex = g0x + ub * dlx, ey = g0y + ub * dly;
        E = fmaxf(ex * (g.qa * ex + g.qb * ey) + ey * (g.qb * ex + g.qc * ey), 0.f);
    }
    if (KT == GAUSSIAN) {
        const float P1 = g1x * (g.qa * g1x + g.qb * g1y) + g1y * (g.qb * g1x + g.qc * g1y);
        gauss_integral<GRAD>(P, R, K, P1, E, I, dP, dR, dK);
        db = 0.f;
    } else {
        beta_integral<GRAD>(P, R, K, E, g.beta, I, dP, dR, dK, db);
    }
    return g.rho * I;
}

// ---------------------------------------------------------------------------
// binning
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool tile_hit(float pax, float pay, float pbx, float pby, float r, int tx, int ty) {
    const float cx = (tx + 0.5f) * TILE, cy = (ty + 0.5f) * TILE;
    const float dx = pbx - pax, dy = pby - pay;
    const float L2 = dx * dx + dy * dy;
    float t = L2 > 1e-12f ? ((cx - pax) * dx + (cy - pay) * dy) / L2 : 0.f;
    t = fminf(fmaxf(t, 0.f), 1.f);
    const float ex = cx - (pax + t * dx), ey = cy - (pay + t * dy);
    const float rr = r + 0.70710678f * TILE;
    return ex * ex + ey * ey <= rr * rr;
}

__device__ __forceinline__ void tile_rect(float pax, float pay, float pbx, float pby, float ra, float rb, int gx,
                                          int gy, int& x0, int& y0, int& x1, int& y1) {
    float xmin = fminf(pax - ra, pbx - rb), xmax = fmaxf(pax + ra, pbx + rb);
    float ymin = fminf(pay - ra, pby - rb), ymax = fmaxf(pay + ra, pby + rb);
    const float lim = 1e6f;
    xmin = fminf(fmaxf(xmin, -lim), lim);
    xmax = fminf(fmaxf(xmax, -lim), lim);
    ymin = fminf(fmaxf(ymin, -lim), lim);
    ymax = fminf(fmaxf(ymax, -lim), lim);
    x0 = max(0, (int)floorf(xmin / TILE));
    y0 = max(0, (int)floorf(ymin / TILE));
    x1 = min(gx, (int)floorf(xmax / TILE) + 1);
    y1 = min(gy, (int)floorf(ymax / TILE) + 1);
}

__global__ void count_tiles_kernel(int N, const float* __restrict__ pa, const float* __restrict__ pb,
                                   const float* __restrict__ rad, int gx, int gy, int* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float ra = rad[2 * i], rb = rad[2 * i + 1];
    if (!(ra > 0.f) || !(rb > 0.f)) {
        counts[i] = 0;
        return;
    }
    const float pax = pa[2 * i], pay = pa[2 * i + 1], pbx = pb[2 * i], pby = pb[2 * i + 1];
    int x0, y0, x1, y1;
    tile_rect(pax, pay, pbx, pby, ra, rb, gx, gy, x0, y0, x1, y1);
    const float r = fmaxf(ra, rb);
    int c = 0;
    for (int ty = y0; ty < y1; ++ty)
        for (int tx = x0; tx < x1; ++tx) c += tile_hit(pax, pay, pbx, pby, r, tx, ty);
    counts[i] = c;
}

__global__ void emit_keys_kernel(int N, const float* __restrict__ pa, const float* __restrict__ pb,
                                 const float* __restrict__ rad, const float* __restrict__ depth, int gx, int gy,
                                 const int64_t* __restrict__ offsets, const int* __restrict__ counts,
                                 int64_t* __restrict__ keys, int32_t* __restrict__ vals) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float ra = rad[2 * i], rb = rad[2 * i + 1];
    if (!(ra > 0.f) || !(rb > 0.f)) return;
    const float pax = pa[2 * i], pay = pa[2 * i + 1], pbx = pb[2 * i], pby = pb[2 * i + 1];
    int x0, y0, x1, y1;
    tile_rect(pax, pay, pbx, pby, ra, rb, gx, gy, x0, y0, x1, y1);
    const float r = fmaxf(ra, rb);
    int64_t off = offsets[i];
    const int64_t end = off + counts[i];  // never write more than counted (robust to fp differences)
    const uint32_t dbits = __float_as_uint(fmaxf(depth[i], 0.f));
    for (int ty = y0; ty < y1 && off < end; ++ty)
        for (int tx = x0; tx < x1 && off < end; ++tx)
            if (tile_hit(pax, pay, pbx, pby, r, tx, ty)) {
                keys[off] = ((int64_t)(ty * gx + tx) << 32) | (int64_t)dbits;
                vals[off] = i;
                ++off;
            }
}

// keys of unused slots hold the sentinel tile id n_tiles and sort to the end; they are ignored here.
__global__ void tile_ranges_kernel(int64_t M, int n_tiles, const int64_t* __restrict__ keys, int2* __restrict__ ranges) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    const int tile = (int)(keys[idx] >> 32);
    const int prev = idx == 0 ? -1 : (int)(keys[idx - 1] >> 32);
    const int next = idx == M - 1 ? n_tiles : (int)(keys[idx + 1] >> 32);
    if (tile >= n_tiles) return;
    if (prev != tile) ranges[tile].x = (int)idx;
    if (next != tile) ranges[tile].y = (int)idx + 1;
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

__device__ __forceinline__ void load_prim(Prim& s, int id, const float* pa, const float* pb, const float* ca,
                                          const float* cb, const float* Q, const float* rho, const float* col,
                                          const float* beta) {
    s.pax = pa[2 * id];
    s.pay = pa[2 * id + 1];
    s.pbx = pb[2 * id];
    s.pby = pb[2 * id + 1];
    s.ca = ca[id];
    s.cb = cb[id];
    s.qa = Q[3 * id];
    s.qb = Q[3 * id + 1];
    s.qc = Q[3 * id + 2];
    s.rho = rho[id];
    s.beta = beta ? beta[id] : 0.f;
#pragma unroll
    for (int c = 0; c < NCH; ++c) s.col[c] = col[NCH * id + c];
}

template <int KT>
__global__ void __launch_bounds__(BLOCK_SIZE)
    raster_fwd_kernel(int W, int H, int gx, const int2* __restrict__ ranges, const int32_t* __restrict__ plist,
                      const float* __restrict__ pa, const float* __restrict__ pb, const float* __restrict__ ca,
                      const float* __restrict__ cb, const float* __restrict__ Q, const float* __restrict__ rho,
                      const float* __restrict__ col, const float* __restrict__ beta, const float* __restrict__ bg,
                      float alpha_min, float T_min, float* __restrict__ out_rgb, float* __restrict__ out_alpha,
                      float* __restrict__ out_T, int32_t* __restrict__ out_n) {
    auto block = cg::this_thread_block();
    const int tile_id = blockIdx.y * gx + blockIdx.x;
    const int px = blockIdx.x * TILE + threadIdx.x, py = blockIdx.y * TILE + threadIdx.y;
    const bool inside = px < W && py < H;
    const float fx = px + 0.5f, fy = py + 0.5f;
    const int2 range = ranges[tile_id];
    const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int todo = range.y - range.x;

    __shared__ Prim sh[BLOCK_SIZE];
    bool done = !inside;
    float T = 1.f;
    float C[NCH] = {0.f};
    int contributor = 0, last = 0;
    float I, dP, dR, dK, db, g0x, g0y, dlx, dly;

    for (int r = 0; r < rounds; ++r, todo -= BLOCK_SIZE) {
        if (__syncthreads_count(done) == BLOCK_SIZE) break;
        const int progress = r * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) load_prim(sh[block.thread_rank()], plist[range.x + progress], pa, pb, ca, cb, Q, rho, col, beta);
        block.sync();
        const int n = min(BLOCK_SIZE, todo);
        for (int j = 0; !done && j < n; ++j) {
            ++contributor;
            const float tau = eval_tau<KT, false>(sh[j], fx, fy, I, dP, dR, dK, db, g0x, g0y, dlx, dly);
            const float a_raw = 1.f - expf(-tau);
            const float alpha = (a_raw > 0.99f) ? 0.99f : a_raw;  // NaN stays NaN (fminf would drop it)
            if (!(alpha >= alpha_min) || !(alpha > 0.f)) continue;
            const float tT = T * (1.f - alpha);
            if (tT < T_min) {
                done = true;
                continue;
            }
#pragma unroll
            for (int c = 0; c < NCH; ++c) C[c] += sh[j].col[c] * alpha * T;
            T = tT;
            last = contributor;
        }
    }
    if (inside) {
        const int pix = py * W + px;
        for (int c = 0; c < NCH; ++c) out_rgb[NCH * pix + c] = C[c] + T * bg[c];
        out_alpha[pix] = 1.f - T;
        out_T[pix] = T;
        out_n[pix] = last;
    }
}

// ---------------------------------------------------------------------------
// backward
// ---------------------------------------------------------------------------

template <int KT>
__global__ void __launch_bounds__(BLOCK_SIZE)
    raster_bwd_kernel(int W, int H, int gx, const int2* __restrict__ ranges, const int32_t* __restrict__ plist,
                      const float* __restrict__ pa, const float* __restrict__ pb, const float* __restrict__ ca,
                      const float* __restrict__ cb, const float* __restrict__ Q, const float* __restrict__ rho,
                      const float* __restrict__ col, const float* __restrict__ beta, const float* __restrict__ bg,
                      float alpha_min, const float* __restrict__ fin_T, const int32_t* __restrict__ fin_n,
                      const float* __restrict__ dL_drgb, const float* __restrict__ dL_dalpha,
                      float* __restrict__ g_pa, float* __restrict__ g_pb, float* __restrict__ g_ca,
                      float* __restrict__ g_cb, float* __restrict__ g_Q, float* __restrict__ g_rho,
                      float* __restrict__ g_col, float* __restrict__ g_beta) {
    auto block = cg::this_thread_block();
    const int tile_id = blockIdx.y * gx + blockIdx.x;
    const int px = blockIdx.x * TILE + threadIdx.x, py = blockIdx.y * TILE + threadIdx.y;
    const bool inside = px < W && py < H;
    const int pix = inside ? py * W + px : 0;
    const float fx = px + 0.5f, fy = py + 0.5f;
    const int2 range = ranges[tile_id];
    const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int todo = range.y - range.x;

    __shared__ Prim sh[BLOCK_SIZE];
    __shared__ int sh_id[BLOCK_SIZE];

    const float T_final = inside ? fin_T[pix] : 0.f;
    float T = T_final;
    const int last = inside ? fin_n[pix] : 0;
    float dpix[NCH], bgdot = 0.f;
    for (int c = 0; c < NCH; ++c) {
        dpix[c] = inside ? dL_drgb[NCH * pix + c] : 0.f;
        bgdot += bg[c] * dpix[c];
    }
    const float dA_out = inside ? dL_dalpha[pix] : 0.f;
    float accum[NCH] = {0.f}, last_col[NCH] = {0.f};
    float last_alpha = 0.f;
    int contributor = todo;
    const int lane = block.thread_rank() % 32;

    for (int r = 0; r < rounds; ++r, todo -= BLOCK_SIZE) {
        block.sync();
        const int progress = r * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            const int id = plist[range.y - progress - 1];
            sh_id[block.thread_rank()] = id;
            load_prim(sh[block.thread_rank()], id, pa, pb, ca, cb, Q, rho, col, beta);
        }
        block.sync();
        const int n = min(BLOCK_SIZE, todo);
        for (int j = 0; j < n; ++j) {
            --contributor;
            float gr[NGRAD];
#pragma unroll
            for (int k = 0; k < NGRAD; ++k) gr[k] = 0.f;
            bool valid = inside && contributor < last;
            if (valid) {
                const Prim& g = sh[j];
                float I, dP, dR, dK, db, g0x, g0y, dlx, dly;
                const float tau = eval_tau<KT, true>(g, fx, fy, I, dP, dR, dK, db, g0x, g0y, dlx, dly);
                const float a_raw = 1.f - expf(-tau);
                const float alpha = (a_raw > 0.99f) ? 0.99f : a_raw;
                if (!(alpha >= alpha_min) || !(alpha > 0.f)) {
                    valid = false;
                } else {
                    T = T / (1.f - alpha);
                    const float w = alpha * T;
                    float dL_da = 0.f;
#pragma unroll
                    for (int c = 0; c < NCH; ++c) {
                        accum[c] = last_alpha * last_col[c] + (1.f - last_alpha) * accum[c];
                        last_col[c] = g.col[c];
                        dL_da += (g.col[c] - accum[c]) * dpix[c];
                        gr[10 + c] = w * dpix[c];
                    }
                    dL_da *= T;
                    last_alpha = alpha;
                    dL_da += (-T_final / (1.f - alpha)) * bgdot;
                    dL_da += (T_final / (1.f - alpha)) * dA_out;
                    const float dtau = (a_raw < 0.99f) ? dL_da * (1.f - a_raw) : 0.f;
                    gr[9] = dtau * I;  // rho
                    const float dI = dtau * g.rho;
                    const float gP = dI * dP, gR = dI * dR, gK = dI * dK;
                    gr[13] = dI * db;  // beta
                    // P = g0'Qg0, R = dl'Qg0, K = dl'Qdl
                    const float Qg0x = g.qa * g0x + g.qb * g0y, Qg0y = g.qb * g0x + g.qc * g0y;
                    const float Qdlx = g.qa * dlx + g.qb * dly, Qdly = g.qb * dlx + g.qc * dly;
                    const float dg0x = 2.f * gP * Qg0x + gR * Qdlx, dg0y = 2.f * gP * Qg0y + gR * Qdly;
                    const float ddlx = gR * Qg0x + 2.f * gK * Qdlx, ddly = gR * Qg0y + 2.f * gK * Qdly;
                    gr[6] = gP * g0x * g0x + gR * g0x * dlx + gK * dlx * dlx;
                    gr[7] = 2.f * gP * g0x * g0y + gR * (g0x * dly + g0y * dlx) + 2.f * gK * dlx * dly;
                    gr[8] = gP * g0y * g0y + gR * g0y * dly + gK * dly * dly;
                    // dl = g1 - g0 ; g0 = (p - pa) ca ; g1 = (p - pb) cb
                    const float t0x = dg0x - ddlx, t0y = dg0y - ddly;
                    gr[0] = -g.ca * t0x;
                    gr[1] = -g.ca * t0y;
                    gr[2] = -g.cb * ddlx;
                    gr[3] = -g.cb * ddly;
                    gr[4] = (fx - g.pax) * t0x + (fy - g.pay) * t0y;
                    gr[5] = (fx - g.pbx) * ddlx + (fy - g.pby) * ddly;
                }
            }
            if (__any_sync(0xffffffffu, valid)) {
#pragma unroll
                for (int k = 0; k < NGRAD; ++k)
                    for (int off = 16; off > 0; off >>= 1) gr[k] += __shfl_down_sync(0xffffffffu, gr[k], off);
                if (lane == 0) {
                    const int id = sh_id[j];
                    atomicAdd(&g_pa[2 * id], gr[0]);
                    atomicAdd(&g_pa[2 * id + 1], gr[1]);
                    atomicAdd(&g_pb[2 * id], gr[2]);
                    atomicAdd(&g_pb[2 * id + 1], gr[3]);
                    atomicAdd(&g_ca[id], gr[4]);
                    atomicAdd(&g_cb[id], gr[5]);
                    atomicAdd(&g_Q[3 * id], gr[6]);
                    atomicAdd(&g_Q[3 * id + 1], gr[7]);
                    atomicAdd(&g_Q[3 * id + 2], gr[8]);
                    atomicAdd(&g_rho[id], gr[9]);
                    for (int c = 0; c < NCH; ++c) atomicAdd(&g_col[NCH * id + c], gr[10 + c]);
                    if (KT == BETA) atomicAdd(&g_beta[id], gr[13]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// host wrappers
// ---------------------------------------------------------------------------

#define CHECK_F(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32, #x " must be contiguous float32 CUDA")

std::vector<torch::Tensor> bin_segments(torch::Tensor pa, torch::Tensor pb, torch::Tensor rad, torch::Tensor depth,
                                        int W, int H) {
    CHECK_F(pa);
    CHECK_F(pb);
    CHECK_F(rad);
    CHECK_F(depth);
    const at::cuda::OptionalCUDAGuard guard(pa.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int N = pa.size(0);
    const int gx = (W + TILE - 1) / TILE, gy = (H + TILE - 1) / TILE;
    auto opt_i = pa.options().dtype(torch::kInt32);
    auto opt_l = pa.options().dtype(torch::kInt64);
    auto counts = torch::zeros({N}, opt_i);
    auto ranges = torch::zeros({gx * gy, 2}, opt_i);
    if (N == 0) return {ranges, torch::zeros({0}, opt_i), counts};
    count_tiles_kernel<<<(N + 255) / 256, 256, 0, stream>>>(N, pa.data_ptr<float>(), pb.data_ptr<float>(),
                                                            rad.data_ptr<float>(), gx, gy, counts.data_ptr<int>());
    auto csum = torch::cumsum(counts, 0, torch::kInt64);
    const int64_t M = csum[-1].item<int64_t>();
    if (M == 0) return {ranges, torch::zeros({0}, opt_i), counts};
    auto offsets = csum - counts.to(torch::kInt64);
    auto keys = torch::full({M}, (int64_t)(gx * gy) << 32, opt_l);  // sentinel tile
    auto vals = torch::zeros({M}, opt_i);
    emit_keys_kernel<<<(N + 255) / 256, 256, 0, stream>>>(N, pa.data_ptr<float>(), pb.data_ptr<float>(),
                                                          rad.data_ptr<float>(), depth.data_ptr<float>(), gx, gy,
                                                          offsets.data_ptr<int64_t>(), counts.data_ptr<int>(),
                                                          keys.data_ptr<int64_t>(), vals.data_ptr<int32_t>());
    auto sorted = torch::sort(keys);
    auto skeys = std::get<0>(sorted).contiguous();
    auto plist = vals.index_select(0, std::get<1>(sorted)).contiguous();
    tile_ranges_kernel<<<(M + 255) / 256, 256, 0, stream>>>(M, gx * gy, skeys.data_ptr<int64_t>(),
                                                            reinterpret_cast<int2*>(ranges.data_ptr<int>()));
    return {ranges, plist, counts};
}

std::vector<torch::Tensor> rasterize_fwd(torch::Tensor ranges, torch::Tensor plist, torch::Tensor pa,
                                         torch::Tensor pb, torch::Tensor ca, torch::Tensor cb, torch::Tensor Q,
                                         torch::Tensor rho, torch::Tensor col, torch::Tensor beta, torch::Tensor bg,
                                         int W, int H, int kernel, double alpha_min, double T_min) {
    CHECK_F(pa);
    CHECK_F(pb);
    CHECK_F(ca);
    CHECK_F(cb);
    CHECK_F(Q);
    CHECK_F(rho);
    CHECK_F(col);
    CHECK_F(bg);
    TORCH_CHECK(T_min > 1e-30, "T_min must be > 1e-30 (T is recovered by division in backward)");
    const at::cuda::OptionalCUDAGuard guard(pa.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int gx = (W + TILE - 1) / TILE, gy = (H + TILE - 1) / TILE;
    auto opt = pa.options();
    auto rgb = torch::empty({H, W, NCH}, opt);
    auto alpha = torch::empty({H, W}, opt);
    auto finT = torch::empty({H, W}, opt);
    auto finN = torch::empty({H, W}, opt.dtype(torch::kInt32));
    const float* bptr = (kernel == BETA) ? beta.data_ptr<float>() : nullptr;
    dim3 grid(gx, gy), blk(TILE, TILE);
#define LAUNCH_FWD(KT)                                                                                          \
    raster_fwd_kernel<KT><<<grid, blk, 0, stream>>>(                                                            \
        W, H, gx, reinterpret_cast<const int2*>(ranges.data_ptr<int>()), plist.data_ptr<int32_t>(),             \
        pa.data_ptr<float>(), pb.data_ptr<float>(), ca.data_ptr<float>(), cb.data_ptr<float>(),                 \
        Q.data_ptr<float>(), rho.data_ptr<float>(), col.data_ptr<float>(), bptr, bg.data_ptr<float>(),          \
        (float)alpha_min, (float)T_min, rgb.data_ptr<float>(), alpha.data_ptr<float>(), finT.data_ptr<float>(), \
        finN.data_ptr<int32_t>())
    if (kernel == BETA)
        LAUNCH_FWD(BETA);
    else
        LAUNCH_FWD(GAUSSIAN);
    return {rgb, alpha, finT, finN};
}

std::vector<torch::Tensor> rasterize_bwd(torch::Tensor ranges, torch::Tensor plist, torch::Tensor pa,
                                         torch::Tensor pb, torch::Tensor ca, torch::Tensor cb, torch::Tensor Q,
                                         torch::Tensor rho, torch::Tensor col, torch::Tensor beta, torch::Tensor bg,
                                         int W, int H, int kernel, double alpha_min, torch::Tensor finT,
                                         torch::Tensor finN, torch::Tensor dL_drgb, torch::Tensor dL_dalpha) {
    CHECK_F(dL_drgb);
    CHECK_F(dL_dalpha);
    const at::cuda::OptionalCUDAGuard guard(pa.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int N = pa.size(0);
    const int gx = (W + TILE - 1) / TILE, gy = (H + TILE - 1) / TILE;
    auto opt = pa.options();
    auto g_pa = torch::zeros({N, 2}, opt), g_pb = torch::zeros({N, 2}, opt);
    auto g_ca = torch::zeros({N}, opt), g_cb = torch::zeros({N}, opt);
    auto g_Q = torch::zeros({N, 3}, opt), g_rho = torch::zeros({N}, opt);
    auto g_col = torch::zeros({N, NCH}, opt), g_beta = torch::zeros({N}, opt);
    const float* bptr = (kernel == BETA) ? beta.data_ptr<float>() : nullptr;
    dim3 grid(gx, gy), blk(TILE, TILE);
#define LAUNCH_BWD(KT)                                                                                      \
    raster_bwd_kernel<KT><<<grid, blk, 0, stream>>>(                                                        \
        W, H, gx, reinterpret_cast<const int2*>(ranges.data_ptr<int>()), plist.data_ptr<int32_t>(),         \
        pa.data_ptr<float>(), pb.data_ptr<float>(), ca.data_ptr<float>(), cb.data_ptr<float>(),             \
        Q.data_ptr<float>(), rho.data_ptr<float>(), col.data_ptr<float>(), bptr, bg.data_ptr<float>(),      \
        (float)alpha_min, finT.data_ptr<float>(), finN.data_ptr<int32_t>(), dL_drgb.data_ptr<float>(),      \
        dL_dalpha.data_ptr<float>(), g_pa.data_ptr<float>(), g_pb.data_ptr<float>(), g_ca.data_ptr<float>(), \
        g_cb.data_ptr<float>(), g_Q.data_ptr<float>(), g_rho.data_ptr<float>(), g_col.data_ptr<float>(),    \
        g_beta.data_ptr<float>())
    if (kernel == BETA)
        LAUNCH_BWD(BETA);
    else
        LAUNCH_BWD(GAUSSIAN);
    return {g_pa, g_pb, g_ca, g_cb, g_Q, g_rho, g_col, g_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bin_segments", &bin_segments);
    m.def("rasterize_fwd", &rasterize_fwd);
    m.def("rasterize_bwd", &rasterize_bwd);
}
