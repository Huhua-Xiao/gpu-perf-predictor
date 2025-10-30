#include "gemm.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include <fstream>
#include <iomanip>

using namespace std;

struct PerfData{
    // static GPU Specs
    string gpu_name;
    int device_id;
    int cc_major, cc_minor; // Compute Capability
    int sm_count; // Streaming Multiprocessor
    long long l2_size_bytes; // l2 cache
    int shared_mem_per_sm; // shared memory for each sm
    long long total_global_mem; // total global memory
    double mem_bandwidth_GBps_est; // estimated bandwidth
    double peak_flops_fp32_GFLOPs_est; // estimate the peak of fp32 GFLOPs 
    int driver_version; //  drive versions
    int cuda_runtime_version; // cuda drive versions
    int max_threads_per_sm; // max threads per sm
    int max_threads_per_block; // max threads per block
    int warp_size; // warp size

    // Kernel Configuration
    int M, N, K; // size of MM
    string precision; // FP32, TF32、FP16
    int repeats;
    int inner_iters;
    
    // Runtime statistics (per-run averaged over inner_iters)
    double time_ms_mean;
    double time_ms_median;
    double time_ms_p95;
    double time_ms_stddev;
    double gflops_mean;
    double gflops_median;
    double gflops_p95;

    // Derived
    double flops_total; // 2*M*N*K
    double io_bytes_theoretical; // (MK+KN+MN)*sizeof(float)
    double gflops_over_peak_mean; // gflops_mean / peak_fp32
    double arithmetic_intensity_FLOPs_per_Byte; // flops / io_bytes_theoretical
    double memory_throughput_GBps;
    double memory_efficiency_pct;
    double compute_efficiency_pct;
};

static void write_csv_header(std::ofstream& f) {
    f << "gpu_name,device_id,cc_major,cc_minor,sm_count,l2_size_bytes,shared_mem_per_sm,total_global_mem,"
      << "mem_bandwidth_GBps_est,peak_flops_fp32_GFLOPs_est,driver_version,cuda_runtime_version,"
      << "max_threads_per_sm,max_threads_per_block,warp_size,"
      << "M,N,K,precision,repeats,inner_iters,"
      << "time_ms_mean,time_ms_median,time_ms_p95,time_ms_stddev,"
      << "gflops_mean,gflops_median,gflops_p95,"
      << "flops_total,io_bytes_theoretical,gflops_over_peak_mean,arithmetic_intensity_FLOPs_per_Byte,"
      << "memory_throughput_GBps,memory_efficiency_pct,compute_efficiency_pct\n";
}

static void write_csv_row(std::ofstream& f, const PerfData& r) {
    f << r.gpu_name << ","
      << r.device_id << ","
      << r.cc_major << ","
      << r.cc_minor << ","
      << r.sm_count << ","
      << r.l2_size_bytes << ","
      << r.shared_mem_per_sm << ","
      << r.total_global_mem << ","
      << std::fixed << std::setprecision(3) << r.mem_bandwidth_GBps_est << ","
      << std::fixed << std::setprecision(3) << r.peak_flops_fp32_GFLOPs_est << ","
      << r.driver_version << ","
      << r.cuda_runtime_version << ","
      << r.max_threads_per_sm << ","
      << r.max_threads_per_block << ","
      << r.warp_size << ","
      << r.M << "," << r.N << "," << r.K << ","
      << r.precision << ","
      << r.repeats << ","
      << r.inner_iters << ","
      << std::fixed << std::setprecision(6) << r.time_ms_mean << ","
      << std::fixed << std::setprecision(6) << r.time_ms_median << ","
      << std::fixed << std::setprecision(6) << r.time_ms_p95 << ","
      << std::fixed << std::setprecision(6) << r.time_ms_stddev << ","
      << std::fixed << std::setprecision(3) << r.gflops_mean << ","
      << std::fixed << std::setprecision(3) << r.gflops_median << ","
      << std::fixed << std::setprecision(3) << r.gflops_p95 << ","
      << std::fixed << std::setprecision(0) << r.flops_total << ","
      << std::fixed << std::setprecision(0) << r.io_bytes_theoretical << ","
      << std::fixed << std::setprecision(6) << r.gflops_over_peak_mean << ","
      << std::fixed << std::setprecision(6) << r.arithmetic_intensity_FLOPs_per_Byte << ","
      << std::fixed << std::setprecision(3) << r.memory_throughput_GBps << ","
      << std::fixed << std::setprecision(3) << r.memory_efficiency_pct << ","
      << std::fixed << std::setprecision(3) << r.compute_efficiency_pct
      << "\n";
}

// memoryClock * busWidth/8 * 2 / 1e9
static double estimate_mem_bw_GBs(const cudaDeviceProp& p) {
    double memClockHz = static_cast<double>(p.memoryClockRate) * 1000.0;
    double busBytes   = static_cast<double>(p.memoryBusWidth) / 8.0;
    double bw = (memClockHz * busBytes * 2.0) / 1e9; // GB/s
    return bw;
}

static int fp32_cores_per_sm_est(int major, int minor) {
    int cc = major * 10 + minor;
    if (cc >= 80) return 128;  // Ampere and newer
    if (cc >= 75) return 64;   // Turing
    if (cc >= 70) return 64;   // Volta
    if (cc >= 60) return 128;  // Pascal
    return 128;
}

static double estimate_peak_fp32_GFLOPs(const cudaDeviceProp& p) {
    int cores = fp32_cores_per_sm_est(p.major, p.minor);
    double sm_hz = p.clockRate * 1000.0; // kHz->Hz
    return static_cast<double>(p.multiProcessorCount) * cores * 2.0 * (sm_hz / 1e9);
}

static void summarize(const std::vector<double>& samples_ms, double& mean, double& median, double& p95, double& stddev) {
    if (samples_ms.empty()) { 
        mean = median = p95 = stddev = 0.0; 
        return; 
    }
    
    // Calculate mean
    double s = 0.0; 
    for (auto x : samples_ms) {
        s += x;
    }
    mean = s / samples_ms.size();

    // Calculate standard deviation
    double sq_sum = 0.0;
    for (auto x : samples_ms) {
        double diff = x - mean;
        sq_sum += diff * diff;
    }
    stddev = sqrt(sq_sum / samples_ms.size());

    // Calculate median and p95
    std::vector<double> v = samples_ms;
    std::sort(v.begin(), v.end());
    median = v[v.size()/2];

    size_t idx95 = static_cast<size_t>(0.95 * (v.size()-1));
    p95 = v[idx95];
}


class Collector {
    vector<PerfData> data;
    
    // static GPU Specs
    string gpu_name;
    int device_id;
    int cc_major, cc_minor; 
    int sm_count;
    long long l2_size_bytes;
    int shared_mem_per_sm;
    long long total_global_mem;
    double mem_bandwidth_GBps_est;
    double peak_flops_fp32_GFLOPs_est;
    int driver_version;
    int cuda_runtime_version;
    int max_threads_per_sm;
    int max_threads_per_block;
    int warp_size;

    void gpu_info() {
        device_id = 0;
        cudaDeviceProp prop;
        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(&prop, device_id);

        gpu_name = prop.name;
        cc_major = prop.major; 
        cc_minor = prop.minor;
        sm_count = prop.multiProcessorCount;
        shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
        l2_size_bytes = static_cast<long long>(prop.l2CacheSize);
        total_global_mem = static_cast<long long>(prop.totalGlobalMem);
        mem_bandwidth_GBps_est = estimate_mem_bw_GBs(prop);
        peak_flops_fp32_GFLOPs_est = estimate_peak_fp32_GFLOPs(prop);
        max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        max_threads_per_block = prop.maxThreadsPerBlock;
        warp_size = prop.warpSize;
        
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&cuda_runtime_version);

        printf("GPU: %s\n", gpu_name.c_str());
        printf("Compute Capability: %d.%d\n", cc_major, cc_minor);
        printf("SM Count: %d\n", sm_count);
        printf("Peak FP32: %.1f GFLOPS\n", peak_flops_fp32_GFLOPs_est);
        printf("Memory Bandwidth: %.1f GB/s\n", mem_bandwidth_GBps_est);
        printf("\n");
    }

public:

    Collector() {
        gpu_info();
    }

    PerfData benchmark(int M, int N, int K, int inner_iters = 100, int repeats = 10) {

        PerfData perf_data;
        perf_data.inner_iters = inner_iters;
        perf_data.repeats = repeats;

        vector<float> h_A(M * K, 1.0f);
        vector<float> h_B(K * N, 1.0f);

        GEMM gemm(M, N, K);
        gemm.host_to_device(h_A.data(), h_B.data());

        // Warm up
        for (int i = 0; i < 10; i++) {
            gemm.run();
        }
        cudaDeviceSynchronize(); 

        vector<double> per_run_ms;
        per_run_ms.reserve(perf_data.repeats); 
        
        // Timing loop
        for (int j = 0; j < perf_data.repeats; j++){
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            for (int i = 0; i < perf_data.inner_iters; i++) {
                gemm.run();
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float per_total_time = 0.f;
            cudaEventElapsedTime(&per_total_time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            double avg_ms = static_cast<double>(per_total_time) / perf_data.inner_iters;
            per_run_ms.push_back(avg_ms);
        }

        // Calculate statistics
        double t_mean_ms=0, t_median_ms=0, t_p95_ms=0, t_stddev_ms=0;
        summarize(per_run_ms, t_mean_ms, t_median_ms, t_p95_ms, t_stddev_ms);

        // Calculate metrics
        const double flops = 2.0 * static_cast<double>(M) * N * K;
        const double io_bytes_theoretical = ((double)M * K + (double)K * N + (double)M * N) * sizeof(float);
        const double AI = flops / io_bytes_theoretical; 

        double gflops_mean   = (flops / 1e9) / (t_mean_ms   / 1000.0);
        double gflops_median = (flops / 1e9) / (t_median_ms / 1000.0);
        double gflops_p95    = (flops / 1e9) / (t_p95_ms    / 1000.0);

        double memory_throughput_GBps = (io_bytes_theoretical / 1e9) / (t_mean_ms / 1000.0);
        double memory_efficiency_pct = (memory_throughput_GBps / mem_bandwidth_GBps_est) * 100.0;
        double compute_efficiency_pct = (gflops_mean / peak_flops_fp32_GFLOPs_est) * 100.0;

        // Fill PerfData structure
        perf_data.gpu_name = gpu_name;
        perf_data.device_id = device_id;
        perf_data.cc_major = cc_major;
        perf_data.cc_minor = cc_minor;
        perf_data.sm_count = sm_count;
        perf_data.l2_size_bytes = l2_size_bytes;
        perf_data.shared_mem_per_sm = shared_mem_per_sm;
        perf_data.total_global_mem = total_global_mem;
        perf_data.mem_bandwidth_GBps_est = mem_bandwidth_GBps_est;
        perf_data.peak_flops_fp32_GFLOPs_est = peak_flops_fp32_GFLOPs_est;
        perf_data.driver_version = driver_version;
        perf_data.cuda_runtime_version = cuda_runtime_version;
        perf_data.max_threads_per_sm = max_threads_per_sm;
        perf_data.max_threads_per_block = max_threads_per_block;
        perf_data.warp_size = warp_size;

        perf_data.M = M;
        perf_data.N = N;
        perf_data.K = K;
        perf_data.precision = "FP32";

        perf_data.time_ms_mean = t_mean_ms;
        perf_data.time_ms_median = t_median_ms;
        perf_data.time_ms_p95 = t_p95_ms;
        perf_data.time_ms_stddev = t_stddev_ms;

        perf_data.gflops_mean = gflops_mean;
        perf_data.gflops_median = gflops_median;
        perf_data.gflops_p95 = gflops_p95;

        perf_data.flops_total = flops;
        perf_data.io_bytes_theoretical = io_bytes_theoretical;
        perf_data.arithmetic_intensity_FLOPs_per_Byte = AI;
        perf_data.gflops_over_peak_mean = (peak_flops_fp32_GFLOPs_est > 0.0) ? 
                                           (gflops_mean / peak_flops_fp32_GFLOPs_est) : 0.0;
        
        perf_data.memory_throughput_GBps = memory_throughput_GBps;
        perf_data.memory_efficiency_pct = memory_efficiency_pct;
        perf_data.compute_efficiency_pct = compute_efficiency_pct;

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

    void save_to_csv(const std::string& filename) {
        std::ifstream fin(filename);
        bool exists = fin.good();
        fin.close();

        std::ofstream f(filename, std::ios::app);
        if (!f.is_open()) {
            printf("Failed to open file: %s\n", filename.c_str());
            return;
        }
        if (!exists) write_csv_header(f);
        for (const auto& d : data) write_csv_row(f, d);
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
