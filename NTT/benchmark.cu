#include "ntt.cu"
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
#include <random>

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
    int N;
    uint32_t modulus;       // modulus p
    uint32_t primitive_root;// primitive root g
    int repeats;
    int inner_iters;
    string algorithm;       // "Custom-NTT"
    
    
    // Runtime statistics (per-run averaged over inner_iters)
    double time_ms_mean;
    double time_ms_median;
    double time_ms_p95;
    double time_ms_stddev;

    // Derived
    double butterflies_total;   // (N/2)*log2(N)
    double modops_total;        // butterflies_total * 3
    double modops_per_sec;      // modops_total / mean_time
    double bytes_total_theoretical; // ~8*N*log2(N)  (per 32-bit data)
    double memory_throughput_GBps;
    double memory_efficiency_pct;   // / mem_bandwidth_GBps_est
};

// Utility functions (estimate bandwidth/peak GFLOPs/statistics)
static double estimate_mem_bw_GBs(const cudaDeviceProp& p) {
    double memClockHz = static_cast<double>(p.memoryClockRate) * 1000.0;
    double busBytes   = static_cast<double>(p.memoryBusWidth) / 8.0;
    return (memClockHz * busBytes * 2.0) / 1e9; // GB/s
}
static int fp32_cores_per_sm_est(int major, int minor) {
    int cc = major * 10 + minor;
    if (cc >= 80) return 128;  // Ampere+
    if (cc >= 75) return 64;   // Turing
    if (cc >= 70) return 64;   // Volta
    if (cc >= 60) return 128;  // Pascal
    return 128;
}
static double estimate_peak_fp32_GFLOPs(const cudaDeviceProp& p) {
    int cores = fp32_cores_per_sm_est(p.major, p.minor);
    double sm_hz = p.clockRate * 1000.0;
    return static_cast<double>(p.multiProcessorCount) * cores * 2.0 * (sm_hz / 1e9);
}
static void summarize(const std::vector<double>& ms, double& mean, double& median, double& p95, double& stddev) {
    if (ms.empty()) { mean=median=p95=stddev=0.0; return; }
    double s=0.0; for (auto x:ms) s+=x; mean = s/ms.size();
    double sq=0.0; for (auto x:ms){ double d=x-mean; sq+=d*d; } stddev = sqrt(sq/ms.size());
    vector<double> v=ms; sort(v.begin(),v.end()); median=v[v.size()/2];
    size_t i95 = (size_t)(0.95*(v.size()-1)); p95 = v[i95];
}

// --- CSV ---
static void write_csv_header(std::ofstream& f) {
    f << "gpu_name,device_id,cc_major,cc_minor,sm_count,l2_size_bytes,shared_mem_per_sm,total_global_mem,"
      << "mem_bandwidth_GBps_est,peak_flops_fp32_GFLOPs_est,driver_version,cuda_runtime_version,"
      << "max_threads_per_sm,max_threads_per_block,warp_size,"
      << "N,modulus,primitive_root,repeats,inner_iters,algorithm,"
      << "time_ms_mean,time_ms_median,time_ms_p95,time_ms_stddev,"
      << "butterflies_total,modops_total,modops_per_sec,"
      << "bytes_total_theoretical,memory_throughput_GBps,memory_efficiency_pct\n";
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
      << r.N << ","
      << r.modulus << ","
      << r.primitive_root << ","
      << r.repeats << ","
      << r.inner_iters << ","
      << r.algorithm << ","
      << std::fixed << std::setprecision(6) << r.time_ms_mean << ","
      << std::fixed << std::setprecision(6) << r.time_ms_median << ","
      << std::fixed << std::setprecision(6) << r.time_ms_p95 << ","
      << std::fixed << std::setprecision(6) << r.time_ms_stddev << ","
      << std::fixed << std::setprecision(0) << r.butterflies_total << ","
      << std::fixed << std::setprecision(0) << r.modops_total << ","
      << std::fixed << std::setprecision(3) << r.modops_per_sec << ","
      << std::fixed << std::setprecision(0) << r.bytes_total_theoretical << ","
      << std::fixed << std::setprecision(3) << r.memory_throughput_GBps << ","
      << std::fixed << std::setprecision(3) << r.memory_efficiency_pct
      << "\n";
}

// --- Collector (NTT) ---

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
    string algorithm;

    void gpu_info(int method) {
        device_id = 0;
        cudaDeviceProp prop;
        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(&prop, device_id);

        gpu_name = prop.name;
        cc_major = prop.major; cc_minor = prop.minor;
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

        if (method == 1) algorithm = "Custom-NTT";
        else { fprintf(stderr, "Invalid method=%d (use 1=Custom-NTT)\n", method); std::exit(1); }

        printf("GPU: %s\n", gpu_name.c_str());
        printf("Algorithm: %s\n", algorithm.c_str());
        printf("Compute Capability: %d.%d\n", cc_major, cc_minor);
        printf("SM Count: %d | Warp Size: %d\n", sm_count, warp_size);
        printf("Peak FP32: %.1f GFLOPS | Mem BW: %.1f GB/s\n\n",
               peak_flops_fp32_GFLOPs_est, mem_bandwidth_GBps_est);
    }

public:

    Collector(int method) {
        gpu_info(method);
    }

    PerfData benchmark_NTT(int N, uint32_t modulus=998244353u, uint32_t primitive_root=3u,
        int inner_iters=50, int repeats=10)
        {
            PerfData r;
            r.N=N; 
            r.modulus=modulus; 
            r.primitive_root=primitive_root;
            r.repeats=repeats; 
            r.inner_iters=inner_iters; 
            r.algorithm="Custom-NTT";   

            // random input (uniform in [0,p-1])
            std::vector<uint32_t> h(N);
            {
                std::mt19937 rng(42);
                std::uniform_int_distribution<uint32_t> dist(0, modulus-1);

                for (int i=0; i<N; ++i) {
                    h[i]=dist(rng);
                }
            }

            NTT ntt(N, 1, modulus, primitive_root);
            ntt.host_to_device(h.data());

            // warmup
            for (int i=0; i<10; ++i) {
                ntt.forward();
            }
            cudaDeviceSynchronize();

            vector<double> per_run_ms; per_run_ms.reserve(repeats);

            for (int rep=0; rep<repeats; ++rep) {
                cudaEvent_t start, stop; 

                cudaEventCreate(&start); 
                cudaEventCreate(&stop);
                cudaEventRecord(start);

                for (int it=0; it<inner_iters; ++it) {
                    ntt.forward(); // timing forward;
                }

                cudaEventRecord(stop); 
                cudaEventSynchronize(stop);

                float total_ms=0.f; 

                cudaEventElapsedTime(&total_ms, start, stop);
                cudaEventDestroy(start); 
                cudaEventDestroy(stop);
                per_run_ms.push_back(double(total_ms)/inner_iters);
            }

            summarize(per_run_ms, r.time_ms_mean, r.time_ms_median, r.time_ms_p95, r.time_ms_stddev);

            // Estimate metrics
            const int logN = (int)std::log2((double)N);
            const double butterflies = (double)N/2.0 * (double)logN;        
            const double modops = butterflies * 3.0;                  
            const double bytes_theory = 8.0 * (double)N * (double)logN;

            r.butterflies_total = butterflies;
            r.modops_total = modops;
            r.modops_per_sec = (r.time_ms_mean>0) ? (modops / (r.time_ms_mean/1000.0)) : 0.0;

            r.bytes_total_theoretical = bytes_theory;
            r.memory_throughput_GBps = (r.time_ms_mean>0) ? ((bytes_theory/1e9) / (r.time_ms_mean/1000.0)) : 0.0;
            r.memory_efficiency_pct = (mem_bandwidth_GBps_est>0) ? (r.memory_throughput_GBps / mem_bandwidth_GBps_est * 100.0) : 0.0;

            r.gpu_name = gpu_name; 
            r.device_id = device_id;
            r.cc_major = cc_major; 
            r.cc_minor = cc_minor;
            r.sm_count = sm_count; 
            r.l2_size_bytes = l2_size_bytes;
            r.shared_mem_per_sm = shared_mem_per_sm; 
            r.total_global_mem = total_global_mem;
            r.mem_bandwidth_GBps_est = mem_bandwidth_GBps_est;
            r.peak_flops_fp32_GFLOPs_est = peak_flops_fp32_GFLOPs_est;
            r.driver_version = driver_version; 
            r.cuda_runtime_version = cuda_runtime_version;
            r.max_threads_per_sm = max_threads_per_sm; 
            r.max_threads_per_block = max_threads_per_block;
            r.warp_size = warp_size;

            return r;
    }

    void differ_size_loop(int num_samples, int N_default=1<<16, int repeats=10, int inner_iters=50) {
        random_device rd; mt19937 gen(rd());
        uniform_real_distribution<> U(0.0, 1.0);

        printf("Starting NTT benchmarking with %d samples...\n", num_samples);
        for (int i=0; i<num_samples; ++i){
            int k = 10 + (int)std::floor(U(gen)*U(gen)* (20-10+1));
            int N = 1 << k;

            data.push_back(benchmark_NTT(N, 998244353u, 3u, inner_iters, repeats));
            cudaDeviceSynchronize();

            if ((i+1) % 1000 == 0) {
                save_to_csv("ntt_benchmark_results.csv");
                data.clear();
                printf("  → Checkpoint saved (%d samples so far)\n\n", i+1);
            }
        }
        printf("Done!\n");
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

        for (const auto& d : data) { 
            write_csv_row(f, d);
        }

        f.close(); 
        printf("Results saved to %s\n", filename.c_str());
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <method> <num_samples> [N] [repeats] [inner_iters]\n", argv[0]);
        printf("  method: 1 = Custom-NTT\n");
        printf("  Examples:\n");
        printf("    %s 1 100            # 100 samples, N randonly take 2^k\n", argv[0]);
        printf("    %s 1 1 65536 10 50  # Single: N=65536, repeats=10, inner_iters=50\n", argv[0]);
        return 1;
    }

    int method = std::stoi(argv[1]);
    int num_samples = std::stoi(argv[2]);
    int N = (argc >= 4) ? std::stoi(argv[3]) : -1;
    int repeats = (argc >= 5) ? std::stoi(argv[4]) : 10;
    int inner_iters = (argc >= 6) ? std::stoi(argv[5]) : 50;

    if (method != 1) { printf("Error: method must be 1 (Custom-NTT)\n"); return 1; }
    if (num_samples <= 0) { printf("Error: num_samples must be > 0\n"); return 1; }

    Collector collector(method);

    if (N > 0) {
        for (int i=0;i<num_samples;++i){
            auto rec = collector.benchmark_NTT(N, 998244353u, 3u, inner_iters, repeats);
            std::vector<PerfData> one{rec};
            std::ifstream fin("ntt_benchmark_results.csv"); 
            bool exists = fin.good(); fin.close();
            std::ofstream f("ntt_benchmark_results.csv", std::ios::app);
            if (!f.is_open()) { printf("Failed to open output\n"); return 1; }
            if (!exists) write_csv_header(f);
            write_csv_row(f, rec);
        }
        printf("Done fixed-N runs.\n");
    } 
    else {
        collector.differ_size_loop(num_samples, 1<<16, repeats, inner_iters);
        collector.save_to_csv("ntt_benchmark_results.csv");
    }
    return 0;
}
