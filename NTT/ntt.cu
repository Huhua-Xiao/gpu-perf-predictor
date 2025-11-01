#define NTT_DEMO_MAIN

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

// Error Check
#define CUDA_CHECK(cmd) do { \
  cudaError_t e_ = (cmd); \
  if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    std::exit(1); \
  } \
} while(0)

// Device-side mode ops
__device__ __forceinline__ uint32_t add_mod(uint32_t a, uint32_t b, uint32_t mod) {
  uint32_t c = a + b;
  if (c >= mod) c -= mod;
  return c;
}

__device__ __forceinline__ uint32_t sub_mod(uint32_t a, uint32_t b, uint32_t mod) {
  return (a >= b) ? (a - b) : (a + mod - b);
}

__device__ __forceinline__ uint32_t mul_mod(uint32_t a, uint32_t b, uint32_t mod) {
  // 64-bit intermediate to avoid overflow
  return (uint64_t(a) * uint64_t(b)) % uint64_t(mod);
}

// Host utilities
static uint32_t mod_pow_u32_host(uint32_t base, uint32_t exp, uint32_t mod) {
    // binary exponentiation）/ power mod
    uint64_t res = 1, b = base % mod;
    while (exp) {
        if (exp & 1u) res = (res * b) % mod;
        b = (b * b) % mod;
        exp >>= 1u;
    }
    return (uint32_t)res;
}

// Extended Euclidean algorithm to find modular inverse
static uint32_t mod_inv_u32(uint32_t a, uint32_t mod) {
    // Extended Euclidean algorithm to find modular inverse
    long long t = 0, newt = 1;
    long long r = (long long)mod, newr = (long long)a;
    while (newr != 0) {
        long long q = r / newr;
        long long tmp = newt; newt = t - q*newt; t = tmp;
        tmp = newr; newr = r - q*newr; r = tmp;
    }
    if (r > 1) return 0; // not invertible
    if (t < 0) t += mod;
    return (uint32_t)t;
}

static bool is_power_of_two(int n) {
    // check if n is a power of two
    return (n >= 2) && ((n & (n - 1)) == 0);
}

// Kernels (bit-reverse & each NTT stage)

// Calculate the bit-reverse index of log2(n)
__device__ __forceinline__ uint32_t bit_reverse(uint32_t x, int logn) {
  // 32-bit reverse then shift
  x = ((x & 0x55555555u) << 1) | ((x & 0xAAAAAAAAu) >> 1);
  x = ((x & 0x33333333u) << 2) | ((x & 0xCCCCCCCCu) >> 2);
  x = ((x & 0x0F0F0F0Fu) << 4) | ((x & 0xF0F0F0F0u) >> 4);
  x = ((x & 0x00FF00FFu) << 8) | ((x & 0xFF00FF00u) >> 8);
  x = (x << 16) | (x >> 16);
  return x >> (32 - logn);
}

// Bit-reverse permutation (in-place, in-place)
__global__ void bitreverse_permute_kernel(uint32_t* a, int n, int logn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t j = bit_reverse((uint32_t)i, logn);
  if (i < (int)j) {
    uint32_t tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }
}

// One NTT stage (DIT)
__global__ void ntt_stage_kernel(uint32_t* a, int n, int m, int step,
                                 const uint32_t* __restrict__ roots, uint32_t mod) {
  // each thread handles one butterfly (one butterfly per thread)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n >> 1; // n/2
  if (tid >= total) return;

  // group size = 2m; inner position k = tid % m; group start g = (tid / m) * 2m
  int k = tid % m;
  int group = (tid / m) * (m << 1);
  int i0 = group + k;
  int i1 = i0 + m;

  uint32_t w = roots[k * step]; // twiddle = root^{k*step}
  uint32_t u = a[i0];
  uint32_t v = mul_mod(a[i1], w, mod);

  uint32_t x = add_mod(u, v, mod);
  uint32_t y = sub_mod(u, v, mod);

  a[i0] = x;
  a[i1] = y;
}

// Scale by constant (for iNTT 1/n normalization)
__global__ void scale_kernel(uint32_t* a, int n, uint32_t c, uint32_t mod) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  a[i] = mul_mod(a[i], c, mod);
}

// NTT class (similar to GEMM style)
class NTT {
public:
  // only custom NTT (method=1) is implemented (self-implemented)
  NTT(int size,
      int method = 1,
      uint32_t modulus = 998244353u,
      uint32_t primitive_root = 3u)
  : algorithm(method), n(size), mod(modulus),
    d_data(nullptr), d_roots_fwd(nullptr), d_roots_inv(nullptr)
  {
    if (!is_power_of_two(n))
      fail("NTT size must be power-of-two >= 2");

    logn = 0; while ((1 << logn) < n) ++logn;

    // inv_n for inverse normalization
    inv_n = mod_inv_u32((uint32_t)n, mod);
    if (!inv_n) fail("modular inverse of n failed");

    // allocate device buffers
    CUDA_CHECK(cudaMalloc((void**)&d_data, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_roots_fwd, (n/2) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_roots_inv, (n/2) * sizeof(uint32_t)));

    // compute omega = primitive_root^((mod-1)/n) mod mod
    uint32_t omega = mod_pow_u32_host(primitive_root, (mod - 1u) / (uint32_t)size, mod);
    uint32_t omega_inv = mod_inv_u32(omega, mod);
    if (!omega || !omega_inv) fail("omega or omega_inv invalid");

    // precompute twiddles: roots[k] = omega^k
    std::vector<uint32_t> h_roots_fwd(n/2), h_roots_inv(n/2);
    {
        uint32_t pw = 1;
        for (int k = 0; k < n/2; ++k) {
            h_roots_fwd[k] = pw;
            pw = (uint64_t)pw * omega % mod;        // forward twiddle
        }
    }
    {
        uint32_t pw = 1;
        for (int k = 0; k < n/2; ++k) {
            h_roots_inv[k] = pw;
            pw = (uint64_t)pw * omega_inv % mod;    // inverse twiddle
        }
    }
    CUDA_CHECK(cudaMemcpy(d_roots_fwd, h_roots_fwd.data(), (n/2)*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_roots_inv, h_roots_inv.data(), (n/2)*sizeof(uint32_t), cudaMemcpyHostToDevice));
  }

  // Copy data from host to device
  void host_to_device(const uint32_t* h_data) {
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  }

  // Copy data from device to host
  void device_to_host(uint32_t* h_data) {
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  }

  // Forward NTT
  void forward() {
    if (algorithm != 1) fail("only custom NTT (method=1) is implemented");

    // 1) Bit-reverse
    launch_bitreverse();

    // 2) logn NTT stages
    uint32_t* roots = d_roots_fwd;
    int threads = 256;
    int total = n >> 1;         // half of n
    int blocks = (total + threads - 1) / threads;

    for (int s = 0; s < logn; ++s) {
      int m  = 1 << s;          // half size
      int step = n >> (s + 1);    // twiddle step (root exponent stride)
      ntt_stage_kernel<<<blocks, threads>>>(d_data, n, m, step, roots, mod);
      CUDA_CHECK(cudaPeekAtLastError()); // check for errors
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Inverse NTT
  void inverse() {
    if (algorithm != 1) fail("only custom NTT (method=1) is implemented");

    // 1) Bit-reverse
    launch_bitreverse();

    // 2) logn NTT stages (using inverse twiddles)
    uint32_t* roots = d_roots_inv;
    int threads = 256;
    int total = n >> 1;         // half of n
    int blocks = (total + threads - 1) / threads;

    for (int s = 0; s < logn; ++s) {
      int m    = 1 << s;
      int step = n >> (s + 1);
      ntt_stage_kernel<<<blocks, threads>>>(d_data, n, m, step, roots, mod);
      CUDA_CHECK(cudaPeekAtLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3) Scale by inv_n (normalization)
    blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(d_data, n, inv_n, mod);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Device pointer (for your own benchmark script)
  uint32_t* device_ptr() const { return d_data; }
  int size() const { return n; }
  uint32_t modulus() const { return mod; }

  ~NTT() {
    if (d_data)       cudaFree(d_data);
    if (d_roots_fwd)  cudaFree(d_roots_fwd);
    if (d_roots_inv)  cudaFree(d_roots_inv);
  }

private:
  int algorithm;
  int n, logn;
  uint32_t mod, inv_n;

  uint32_t* d_data;
  uint32_t* d_roots_fwd;
  uint32_t* d_roots_inv;

  void fail(const char* msg) {
    fprintf(stderr, "[NTT] %s\n", msg);
    std::exit(1);
  }

  void launch_bitreverse() {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    bitreverse_permute_kernel<<<blocks, threads>>>(d_data, n, logn);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};