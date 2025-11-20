# Compiler and flags
SHELL := /bin/bash
NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17
CUDA_LIBS := -lcublas -lnvidia-ml
TARGET := benchmark

# Load CUDA module
# MODULE_LOAD := module load cuda-12.4;

# Source files
SRC := benchmark.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	@$(NVCC) $(NVCCFLAGS) $(SRC) $(CUDA_LIBS) -o $(TARGET)
	@echo "Build complete: ./$(TARGET) <method> <num_samples> <seed --optional>"

clean:
	rm -f $(TARGET)
	@echo "Cleaned build"