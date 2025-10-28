#include "gemm.cu"
#include <fstream>
#include <vector>
#include <string>
#include <tuple>

using namespace std;

struct PerfData{
    string gpu_name;
    int sm_count;
    int M, N, K;
    float time_ms;
    float gflops;
    float bandwidth_gb_s;

    // ... other performance metrics might need to be added
};

class Collector {
    vector<PerfData> data;
    string gpu_name;
    int sm_count;

    void gpu_info() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        gpu_name = prop.name;
        sm_count = prop.multiProcessorCount;
    }

public:

    Collector() {
        gpu_info();
    }

    PerfData benchmark(int M, int N, int K) {

        vector<float> h_A(M * K, 1.0f);
        vector<float> h_B(K * N, 1.0f);

        GEMM gemm(M, N, K);
        gemm.host_to_device(h_A.data(), h_B.data());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        // printf("Warming up the machine...\n");
        // warm up the machine
        for (int i = 0; i < 10; i++) {
            gemm.run();
        }
        // make sure the device synchronize
        cudaDeviceSynchronize(); 
        // printf("Machine warmed up\n");
        
        // begin timing
        // printf("Starting benchmark...\n");
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            gemm.run();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // printf("Benchmarking completed\n");

        float total_time_kernel;
        cudaEventElapsedTime(&total_time_kernel, start, stop);
        // average time per run in ms
        float time = total_time_kernel / 100; 

        float flops = 2.0f * M * N * K;
        float gflops = (flops / 1e9) / (time / 1000.0f);

        float bytes = (M * K + K * N + M * N) * sizeof(float);
        float bandwidth_gb_s = (bytes / 1e9) / (time / 1000.0f);

        PerfData perf_data;
        perf_data.gpu_name = gpu_name;
        perf_data.sm_count = sm_count;
        perf_data.M = M;
        perf_data.N = N;
        perf_data.K = K;
        perf_data.time_ms = time;
        perf_data.bandwidth_gb_s = bandwidth_gb_s;
        perf_data.gflops = gflops;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        return perf_data;
    }

    void differ_size_loop() {
        vector<tuple<int, int, int>> sizes = {
            {512, 512, 512},
            {1024, 1024, 1024},
            {2048, 2048, 2048},
            {4096, 4096, 4096},
            {8192, 8192, 8192}
        };

        for (auto [M, N, K] : sizes) {
            printf("Benchmarking %dx%dx%d...\n", M, N, K);
            data.push_back(benchmark(M, N, K));
        }
    }

    void save_to_csv (std::string filename) {
        ofstream f(filename);  

        // check if the file is open
        if (!f.is_open()) {
            printf("Failed to open file: %s\n", filename.c_str());
            return;
        }    

        f << "gpu_name,sm_count,M,N,K,time_ms,gflops,bandwidth_gb_s\n";

        for(auto& d : data) {
            f << d.gpu_name << "," << d.sm_count << ","
              << d.M << "," << d.N << "," << d.K << ","
              << d.time_ms << "," << d.gflops << ","
              << d.bandwidth_gb_s << "\n";
        }

        f.close();
        printf("Results saved to %s\n", filename.c_str());
    }
};

int main() {
    Collector collector;
    collector.differ_size_loop();
    collector.save_to_csv("benchmark_results.csv");
    return 0;
}