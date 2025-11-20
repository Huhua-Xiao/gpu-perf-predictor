// Functional correctness test for NTT implementation in ntt.cu
// using fixed vectors:
//   a = [1, 2, 3, 4]
//   b = [5, 6, 7, 8]

#include "../ntt.cu"

#include <cstdint>
#include <cstdio>
#include <vector>
#include <cassert>

// ---------------------- CPU-side helpers ----------------------

static inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t mod) {
    uint64_t s = (uint64_t)a + b;
    if (s >= mod) s -= mod;
    return (uint32_t)s;
}

static inline uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t mod) {
    uint64_t p = (uint64_t)a * b;
    return (uint32_t)(p % mod);
}

// naive cyclic convolution (CPU reference)
// c[k] = sum_{i+j ≡ k (mod n)} a[i] * b[j] (mod mod)
static void naive_cyclic_convolution(const std::vector<uint32_t> &a,
                                     const std::vector<uint32_t> &b,
                                     std::vector<uint32_t> &c,
                                     uint32_t mod) {
    int n = (int)a.size();
    assert((int)b.size() == n);
    c.assign(n, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int k = i + j;
            if (k >= n) k -= n; // wrap
            uint32_t prod = mod_mul(a[i], b[j], mod);
            c[k] = mod_add(c[k], prod, mod);
        }
    }
}

// ---------------------- Main test ----------------------

int main() {
    const int n = 4;  // length of our fixed vectors
    printf("Creating NTT instance with N = %d\n", n);

    NTT ntt(n);
    uint32_t mod = ntt.modulus();

    // Fixed input vectors
    std::vector<uint32_t> a = {1, 2, 3, 4};
    std::vector<uint32_t> b = {5, 6, 7, 8};

    printf("Input a: ");
    for (int i = 0; i < n; ++i) printf("%u ", a[i]);
    printf("\n");

    printf("Input b: ");
    for (int i = 0; i < n; ++i) printf("%u ", b[i]);
    printf("\n");

    std::vector<uint32_t> A_freq(n), B_freq(n);
    std::vector<uint32_t> C_freq(n);
    std::vector<uint32_t> c_gpu(n), c_cpu(n);

    // ---- Forward NTT(a) ----
    ntt.host_to_device(a.data());
    ntt.forward();
    ntt.device_to_host(A_freq.data());

    printf("NTT(a): ");
    for (int i = 0; i < n; ++i) printf("%u ", A_freq[i]);
    printf("\n");

    // ---- Forward NTT(b) ----
    ntt.host_to_device(b.data());
    ntt.forward();
    ntt.device_to_host(B_freq.data());

    printf("NTT(b): ");
    for (int i = 0; i < n; ++i) printf("%u ", B_freq[i]);
    printf("\n");

    // ---- Pointwise multiply in frequency domain ----
    for (int i = 0; i < n; ++i) {
        C_freq[i] = mod_mul(A_freq[i], B_freq[i], mod);
    }

    // ---- Inverse NTT(C_freq) -> c_gpu (cyclic convolution) ----
    ntt.host_to_device(C_freq.data());
    ntt.inverse();
    ntt.device_to_host(c_gpu.data());

    // ---- CPU naive cyclic convolution ----
    naive_cyclic_convolution(a, b, c_cpu, mod);

    printf("Convolution (NTT-based) c_gpu: ");
    for (int i = 0; i < n; ++i) printf("%u ", c_gpu[i]);
    printf("\n");

    printf("Convolution (CPU naive) c_cpu: ");
    for (int i = 0; i < n; ++i) printf("%u ", c_cpu[i]);
    printf("\n");

    // ---- Compare results ----
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (c_gpu[i] != c_cpu[i]) {
            fprintf(stderr,
                    "Mismatch at index %d: gpu=%u cpu=%u\n",
                    i, c_gpu[i], c_cpu[i]);
            ok = false;
        }
    }

    if (ok) {
        printf("Convolution results MATCH. Test PASSED.\n");
        return 0;
    } else {
        printf("Convolution results DO NOT MATCH. Test FAILED.\n");
        return 1;
    }
}
