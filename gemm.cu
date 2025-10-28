#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>

class GEMM {
    cublasHandle_t handle;
    float *d_A, *d_B, *d_C;
    int M, N, K;

public:
    GEMM(int m, int n, int k) : 
        M(m), N(n), K(k), d_A(nullptr), d_B(nullptr), d_C(nullptr), handle(nullptr) {

        cublasCreate(&handle);

        cudaMalloc((void**)&d_A, M * K * sizeof(float));
        if (!d_A) {
            printf("cannot allocated array d_A of %d elements\n", M * K);
            exit(1);
        }
        cudaMalloc((void**)&d_B, K * N * sizeof(float));
        if (!d_B) {
            printf("cannot allocated array d_B of %d elements\n", K * N);
            exit(1);
        }
        cudaMalloc((void**)&d_C, M * N * sizeof(float));
        if (!d_C) {
            printf("cannot allocated array d_C of %d elements\n", M * N);
            exit(1);
        }
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
        cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    ~GEMM() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        if (handle) cublasDestroy(handle);
    }
};