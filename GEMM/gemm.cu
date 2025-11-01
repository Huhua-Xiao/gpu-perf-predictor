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


static constexpr int TILE_WIDTH = 16;
__global__ void matrixMulKernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M, int N, int K);


class GEMM {

private:
    int algorithm; // 0 = cuBLAS, 1 == tiled (custom)
    cublasHandle_t handle;
    float *d_A, *d_B, *d_C;
    int M, N, K;


private:
    void run_cublas() {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K,
                    &beta, d_C, N);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run_tiled() {
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
        matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

public:
    GEMM(int m, int n, int k, int method) : 
        algorithm(method), M(m), N(n), K(k), d_A(nullptr), d_B(nullptr), d_C(nullptr), handle(nullptr) {
        
        CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

        if (algorithm == 0) CUBLAS_CHECK(cublasCreate(&handle));

    }

    void host_to_device(float* h_A, float* h_B) {
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    void run() {
        
        if (algorithm == 0) {
            run_cublas();
        }
        else {
            run_tiled();
        }

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


__global__ void matrixMulKernel(const float* __restrict__ A, 
    const float* __restrict__ B, float* C, int M, int N, int K) {

    // +1 padding → fewer bank conflicts
    __shared__ float As[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH + 1];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y; 

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float acc = 0.0f;
    // ceil(K/tile width)
    const int tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH; 


    for (int t = 0; t < tiles; ++t) {
        int a_col = t * TILE_WIDTH + tx;
        int b_row = t * TILE_WIDTH + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}