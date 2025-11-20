# Compiler and flags
NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17
CUDA_LIBS := -lcublas -lnvidia-ml

# Targets
GEMM_TARGET := gemm_benchmark
NTT_TARGET := ntt_benchmark

# Source files
GEMM_SRC := gemm_benchmark.cu gemm.cu
NTT_SRC := ntt_benchmark.cu ntt.cu

.PHONY: all gemm ntt clean

all: gemm ntt

gemm: $(GEMM_TARGET)

ntt: $(NTT_TARGET)

$(GEMM_TARGET): $(GEMM_SRC)
	$(NVCC) $(NVCCFLAGS) $(GEMM_SRC) $(CUDA_LIBS) -o $(GEMM_TARGET)
	@echo "GEMM benchmark built: ./$(GEMM_TARGET)"

$(NTT_TARGET): $(NTT_SRC)
	$(NVCC) $(NVCCFLAGS) $(NTT_SRC) $(CUDA_LIBS) -o $(NTT_TARGET)
	@echo "NTT benchmark built: ./$(NTT_TARGET)"

clean:
	rm -f $(GEMM_TARGET) $(NTT_TARGET)
	@echo "Cleaned all build files"
