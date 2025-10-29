module avail cude

module load cuda-12.4
nvcc benchmark.cu -o benchmark -lcublas
or 
nvcc -O3 -std=c++17 benchmark.cu -lcublas -o benchmark


Ignore the runner now, not implemented.
Will add the explanation later
