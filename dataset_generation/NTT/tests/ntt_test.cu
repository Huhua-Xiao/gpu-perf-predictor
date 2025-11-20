#include "../ntt.cu"

#include <random>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// ---------------------- CPU-side helpers ----------------------

// modular addition
static inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t mod) {
    uint64_t s = (uint64_t)a + b;
    if (s >= mod) s -= mod;
    return (uint32_t)s;
}

// modular subtraction
static inline uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t mod) {
    return (a >= b) ? (a - b) : (uint32_t)(a + mod - b);
}

// modular multiplication
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
            if (k >= n) k -= n;   // wrap mod n
            uint32_t prod = mod_mul(a[i], b[j], mod);
            c[k] = mod_add(c[k], prod, mod);
        }
    }
}

// ---------------------- Tests ----------------------

// 1) Forward + inverse round-trip
static bool test_roundtrip(NTT &ntt, int num_tests, std::mt19937 &rng) {
    int n          = ntt.size();
    uint32_t mod   = ntt.modulus();
    std::uniform_int_distribution<uint32_t> dist(0, mod - 1);

    std::vector<uint32_t> input(n);
    std::vector<uint32_t> output(n);

    printf("Running %d round-trip tests (forward + inverse)...\n", num_tests);

    for (int t = 0; t < num_tests; ++t) {
        // random input
        for (int i = 0; i < n; ++i) {
            input[i] = dist(rng);
        }

        // push to device, forward, inverse, pull back
        ntt.host_to_device(input.data());
        ntt.forward();
        ntt.inverse();
        ntt.device_to_host(output.data());

        // compare
        for (int i = 0; i < n; ++i) {
            if (output[i] != input[i]) {
                fprintf(stderr,
                        "[Roundtrip FAIL] test=%d index=%d expected=%u got=%u\n",
                        t, i, input[i], output[i]);
                return false;
            }
        }
    }

    printf("Round-trip tests PASSED.\n");
    return true;
}

// 2) Convolution correctness via NTT
// We use NTT to compute cyclic convolution and compare with naive CPU convolution.
static bool test_convolution(NTT &ntt, int num_tests, std::mt19937 &rng) {
    int n          = ntt.size();
    uint32_t mod   = ntt.modulus();
    std::uniform_int_distribution<uint32_t> dist(0, mod - 1);

    std::vector<uint32_t> a(n), b(n);
    std::vector<uint32_t> A_freq(n), B_freq(n), C_freq(n);
    std::vector<uint32_t> c_gpu(n), c_cpu(n);

    printf("Running %d convolution tests using NTT...\n", num_tests);

    for (int t = 0; t < num_tests; ++t) {
        // random inputs
        for (int i = 0; i < n; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }

        // ---- Forward NTT(a) -> A_freq ----
        ntt.host_to_device(a.data());
        ntt.forward();
        ntt.device_to_host(A_freq.data());

        // ---- Forward NTT(b) -> B_freq ----
        ntt.host_to_device(b.data());
        ntt.forward();
        ntt.device_to_host(B_freq.data());

        // ---- Pointwise multiply in frequency domain: C_freq = A_freq * B_freq ----
        for (int i = 0; i < n; ++i) {
            C_freq[i] = mod_mul(A_freq[i], B_freq[i], mod);
        }

        // ---- Inverse NTT(C_freq) -> c_gpu (cyclic convolution result) ----
        ntt.host_to_device(C_freq.data());
        ntt.inverse();
        ntt.device_to_host(c_gpu.data());

        // ---- CPU reference cyclic convolution ----
        naive_cyclic_convolution(a, b, c_cpu, mod);

        // ---- Compare ----
        for (int i = 0; i < n; ++i) {
            if (c_gpu[i] != c_cpu[i]) {
                fprintf(stderr,
                        "[Convolution FAIL] test=%d index=%d "
                        "expected=%u got=%u\n",
                        t, i, c_cpu[i], c_gpu[i]);
                return false;
            }
        }
    }

    printf("Convolution tests PASSED.\n");
    return true;
}

// ---------------------- Main ----------------------

int main(int argc, char** argv) {
    // You can make N configurable; for now pick a reasonable power-of-two.
    int n = 1024;
    if (argc == 2) {
        n = std::atoi(argv[1]);
        if (n <= 1) {
            fprintf(stderr, "Invalid N. Use power-of-two >= 2.\n");
            return 1;
        }
    }

    printf("Creating NTT instance with N = %d\n", n);
    NTT ntt(n);

    std::mt19937 rng(123456u);

    // 1) Roundtrip tests
    if (!test_roundtrip(ntt, /*num_tests=*/20, rng)) {
        fprintf(stderr, "Roundtrip tests FAILED.\n");
        return 1;
    }

    // 2) Convolution tests
    if (!test_convolution(ntt, /*num_tests=*/20, rng)) {
        fprintf(stderr, "Convolution tests FAILED.\n");
        return 1;
    }

    printf("All NTT functional tests PASSED.\n");
    return 0;
}
