#include "gemm.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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
    double mem_bandwidth_GBps_est; // estimated bandwidth
    double peak_flops_fp32_GFLOPs_est; // estimate the peak of fp32 GFLOPs 
    int driver_version; //  drive versions
    int cuda_runtime_version; // cuda drive versions

    // Kernel Configuration
    int M, N, K; // size of MM
    string precision; // FP32, TF32、FP16
    int use_tensor_cores; // if use Tensor Cores or not
    int repeats;
    int inner_iters;
    
    // Runtime statistics (per-run averaged over inner_iters)
    double time_ms_mean;
    double time_ms_median;
    double time_ms_p95;
    double gflops_mean;
    double gflops_median;
    double gflops_p95;

    // Derived
    double flops_total; // 2*M*N*K
    double io_bytes_theoretical; // (MK+KN+MN)*sizeof(float)
    double gflops_over_peak_mean; // gflops_mean / peak_fp32
    double arithmetic_intensity_FLOPs_per_Byte; // flops / io_bytes_theoretical
};

static void write_csv_header(std::ofstream& f) {
    f <<
    "gpu_name,device_id,cc_major,cc_minor,sm_count,l2_size_bytes,shared_mem_per_sm,"
    "mem_bandwidth_GBps_est,peak_flops_fp32_GFLOPs_est,driver_version,cuda_runtime_version,"
    "M,N,K,precision,repeats,inner_iters,"
    "time_ms_mean,time_ms_median,time_ms_p95,"
    "gflops_mean,gflops_median,gflops_p95,"
    "arithmetic_intensity_FLOPs_per_Byte,io_bytes_theoretical,gflops_over_peak_mean\n";
}

static void write_csv_row(std::ofstream& f, const PerfData& r) {
    f << r.gpu_name << ","
      << r.device_id << ","
      << r.cc_major << ","
      << r.cc_minor << ","
      << r.sm_count << ","
      << r.l2_size_bytes << ","
      << r.shared_mem_per_sm << ","
      << std::fixed << std::setprecision(3) << r.mem_bandwidth_GBps_est << ","
      << std::fixed << std::setprecision(3) << r.peak_flops_fp32_GFLOPs_est << ","
      << r.driver_version << ","
      << r.cuda_runtime_version << ","    
      << r.M << "," << r.N << "," << r.K << ","
      << r.precision << ","
      << r.repeats << ","
      << r.inner_iters << ","
      << std::fixed << std::setprecision(6) << r.time_ms_mean << ","
      << std::fixed << std::setprecision(6) << r.time_ms_median << ","
      << std::fixed << std::setprecision(6) << r.time_ms_p95 << ","
      << std::fixed << std::setprecision(3) << r.gflops_mean << ","
      << std::fixed << std::setprecision(3) << r.gflops_median << ","
      << std::fixed << std::setprecision(3) << r.gflops_p95 << ","
      << std::fixed << std::setprecision(6) << r.arithmetic_intensity_FLOPs_per_Byte << ","
      << std::fixed << std::setprecision(0) << r.io_bytes_theoretical << ","
      << std::fixed << std::setprecision(6) << r.gflops_over_peak_mean
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
    if (cc >= 75) return 128;
    return 128;
}

static double estimate_peak_fp32_GFLOPs(const cudaDeviceProp& p) {
    int cores = fp32_cores_per_sm_est(p.major, p.minor);
    double sm_hz = p.clockRate * 1000.0; // kHz->Hz
    return static_cast<double>(p.multiProcessorCount) * cores * 2.0 * (sm_hz / 1e9);
    }

static void summarize(const std::vector<double>& samples_ms, double& mean, double& median, double& p95) {
    if (samples_ms.empty()) { 
        mean = median = p95 = 0.0; return; 
    }
    double s = 0.0; 
    for (auto x : samples_ms) {
        s += x;
    }

    mean = s / samples_ms.size();

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
    double mem_bandwidth_GBps_est;
    double peak_flops_fp32_GFLOPs_est;
    int driver_version;
    int cuda_runtime_version;

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
        mem_bandwidth_GBps_est = estimate_mem_bw_GBs(prop);
        peak_flops_fp32_GFLOPs_est = estimate_peak_fp32_GFLOPs(prop);
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&cuda_runtime_version);

    }

public:

    Collector() {
        gpu_info();
    }

    PerfData benchmark(int M, int N, int K) {

        // edit the data based on your need
        PerfData perf_data;
        perf_data.inner_iters = 100;
        perf_data.repeats = 10;
        perf_data.use_tensor_cores = 0;

        vector<float> h_A(M * K, 1.0f);
        vector<float> h_B(K * N, 1.0f);

        GEMM gemm(M, N, K);
        gemm.host_to_device(h_A.data(), h_B.data());

        // printf("Warming up the machine...\n");
        // warm up the machine
        for (int i = 0; i < 10; i++) {
            gemm.run();
        }
        // make sure the device synchronize
        cudaDeviceSynchronize(); 
        // printf("Machine warmed up\n");

        vector<double> per_run_ms;
        per_run_ms.reserve(perf_data.repeats); 
        
        // begin timing
        // printf("Starting benchmark...\n");
        // Doesn't count time
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

            // calculate the average value 
            double avg_ms = static_cast<double>(per_total_time) / perf_data.inner_iters;
            per_run_ms.push_back(avg_ms);
        }

        // update the mean, medain, p95 values by using summarize function
        double t_mean_ms=0, t_median_ms=0, t_p95_ms=0;
        summarize(per_run_ms, t_mean_ms, t_median_ms, t_p95_ms);


        const double flops = 2.0 * static_cast<double>(M) * N * K;
        const double io_bytes_theoretical = ((double)M * K + (double)K * N + (double)M * N) * sizeof(float);
        // Arithmetic Intensity
        const double AI = flops / io_bytes_theoretical; 

        double gflops_mean   = (flops / 1e9) / (t_mean_ms   / 1000.0);
        double gflops_median = (flops / 1e9) / (t_median_ms / 1000.0);
        double gflops_p95    = (flops / 1e9) / (t_p95_ms    / 1000.0);

        // update the perf_data begin here
        perf_data.gpu_name = gpu_name;
        perf_data.cc_major = cc_major;
        perf_data.cc_minor = cc_minor;
        perf_data.sm_count = sm_count;
        perf_data.l2_size_bytes = l2_size_bytes;
        perf_data.shared_mem_per_sm = shared_mem_per_sm;
        perf_data.mem_bandwidth_GBps_est = mem_bandwidth_GBps_est;
        perf_data.peak_flops_fp32_GFLOPs_est = peak_flops_fp32_GFLOPs_est;

        perf_data.M = M;
        perf_data.N = N;
        perf_data.K = K;
        perf_data.precision = "FP32";
        perf_data.device_id = device_id;

        perf_data.time_ms_mean = t_mean_ms;
        perf_data.time_ms_median = t_median_ms;
        perf_data.time_ms_p95 = t_p95_ms;

        perf_data.gflops_mean = gflops_mean;
        perf_data.gflops_median = gflops_median;
        perf_data.gflops_p95 = gflops_p95;

        perf_data.driver_version = driver_version;
        perf_data.cuda_runtime_version = cuda_runtime_version;

        perf_data.flops_total = flops;
        perf_data.io_bytes_theoretical = io_bytes_theoretical;
        perf_data.arithmetic_intensity_FLOPs_per_Byte = AI;
        perf_data.gflops_over_peak_mean = (peak_flops_fp32_GFLOPs_est > 0.0) ? (gflops_mean / peak_flops_fp32_GFLOPs_est) : 0.0;

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