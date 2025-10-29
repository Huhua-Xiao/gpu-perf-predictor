#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>


// It is much more convenient to check the problem 
// directly like this, without the need to write if condition

#define CUDA_CHECK(cmd)                                                           \
    do {                                                                          \
        cudaError_t e_ = (cmd);                                                   \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(e_));                                      \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(cmd)                                                         \
    do {                                                                          \
        cublasStatus_t s_ = (cmd);                                                \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__,\
                    int(s_));                                                     \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)


class GEMM {
    cublasHandle_t handle;
    float *d_A, *d_B, *d_C;
    int M, N, K;

public:
    GEMM(int m, int n, int k) : 
        M(m), N(n), K(k), d_A(nullptr), d_B(nullptr), d_C(nullptr), handle(nullptr) {
        
        CUBLAS_CHECK(cublasCreate(&handle));
        CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
    }

    void host_to_device(float* h_A, float* h_B) {
        cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    }

    void run() {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K,
                    &beta, d_C, N);
    }

    void device_to_host(float* h_C) {
        CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    ~GEMM() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        if (handle) cublasDestroy(handle);
    }
};