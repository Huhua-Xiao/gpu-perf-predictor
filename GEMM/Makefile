# Compiler and flags
SHELL := /bin/bash
NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17
CUDA_LIBS := -lcublas
TARGET := benchmark

# Load CUDA module
MODULE_LOAD := module load cuda-12.4;

# Source files
SRC := benchmark.cu
HEADERS := gemm.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	@$(MODULE_LOAD) $(NVCC) $(NVCCFLAGS) $(SRC) $(CUDA_LIBS) -o $(TARGET)
	@echo "Build complete: ./$(TARGET) <method> <num_samples>"

clean:
	rm -f $(TARGET)
	@echo "Cleaned build"