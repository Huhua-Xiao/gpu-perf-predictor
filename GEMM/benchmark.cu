#include "gemm.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <random>

using namespace std;

struct PerfData {
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

    int M, N, K;
    string precision;
    int repeats;
    int inner_iters;
    string algorithm;
    int tile_width;
    
    double theoretical_occupancy;
    int blocks_per_sm;
    int grid_x, grid_y;
    int block_x, block_y;
    int registers_per_thread;
    int shared_mem_per_block;
    
    double time_ms_mean;
    double time_ms_median;
    double time_ms_p95;
    double time_ms_stddev;
    double gflops_mean;
    double gflops_median;
    double gflops_p95;

    double ops_total;
    double io_bytes_theoretical;
    double gops_over_peak_mean;
    double arithmetic_intensity_ops_per_byte;
    double memory_throughput_GBps;
    double memory_efficiency_pct;
    double compute_efficiency_pct;
};

static void write_csv_header(std::ofstream& f) {
    f << "gpu_name,device_id,cc_major,cc_minor,sm_count,l2_size_bytes,shared_mem_per_sm,total_global_mem,"
      << "mem_bandwidth_GBps_est,peak_flops_fp32_GFLOPs_est,driver_version,cuda_runtime_version,"
      << "max_threads_per_sm,max_threads_per_block,warp_size,"
      << "M,N,K,precision,repeats,inner_iters,algorithm,tile_width,"
      << "theoretical_occupancy,blocks_per_sm,grid_x,grid_y,block_x,block_y,"
      << "registers_per_thread,shared_mem_per_block,"
      << "time_ms_mean,time_ms_median,time_ms_p95,time_ms_stddev,"
      << "gflops_mean,gflops_median,gflops_p95,"
      << "ops_total,io_bytes_theoretical,gops_over_peak_mean,arithmetic_intensity_ops_per_byte,"
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
      << r.algorithm << ","
      << r.tile_width << ","
      << std::fixed << std::setprecision(6) << r.theoretical_occupancy << ","
      << r.blocks_per_sm << ","
      << r.grid_x << "," << r.grid_y << ","
      << r.block_x << "," << r.block_y << ","
      << r.registers_per_thread << ","
      << r.shared_mem_per_block << ","
      << std::fixed << std::setprecision(6) << r.time_ms_mean << ","
      << std::fixed << std::setprecision(6) << r.time_ms_median << ","
      << std::fixed << std::setprecision(6) << r.time_ms_p95 << ","
      << std::fixed << std::setprecision(6) << r.time_ms_stddev << ","
      << std::fixed << std::setprecision(3) << r.gflops_mean << ","
      << std::fixed << std::setprecision(3) << r.gflops_median << ","
      << std::fixed << std::setprecision(3) << r.gflops_p95 << ","
      << std::fixed << std::setprecision(0) << r.ops_total << ","
      << std::fixed << std::setprecision(0) << r.io_bytes_theoretical << ","
      << std::fixed << std::setprecision(6) << r.gops_over_peak_mean << ","
      << std::fixed << std::setprecision(6) << r.arithmetic_intensity_ops_per_byte << ","
      << std::fixed << std::setprecision(3) << r.memory_throughput_GBps << ","
      << std::fixed << std::setprecision(3) << r.memory_efficiency_pct << ","
      << std::fixed << std::setprecision(3) << r.compute_efficiency_pct
      << "\n";
}

static double estimate_mem_bw_GBs(const cudaDeviceProp& p) {
    double memClockHz = static_cast<double>(p.memoryClockRate) * 1000.0;
    double busBytes = static_cast<double>(p.memoryBusWidth) / 8.0;
    return (memClockHz * busBytes * 2.0) / 1e9;
}

static int fp32_cores_per_sm(int major, int minor) {
    const int cc = major*10 + minor;
    // Kepler
    if (cc >= 30 && cc <= 37) return 192;      // 3.0/3.5/3.7
    // Maxwell
    if (cc >= 50 && cc <= 53) return 128;      // 5.0/5.2/5.3
    // Pascal
    if (cc == 60)           return 64;         // GP100
    if (cc == 61 || cc == 62) return 128;      // GP102/104/106, TX2
    // Volta
    if (cc == 70)           return 64;
    // Turing
    if (cc == 75)           return 64;
    // Ampere / Ada / Hopper / Blackwell
    if (cc == 80 || cc == 86 || cc == 89 || cc == 90) return 128;
    return 128;
}

static double peak_fp32_gflops(const cudaDeviceProp& p) {
    const double ghz = static_cast<double>(p.clockRate) / 1e6;
    return static_cast<double>(p.multiProcessorCount)
         * fp32_cores_per_sm(p.major, p.minor)
         * 2.0 * ghz * 1.2; // FMA and boost
}

static void summarize(const std::vector<double>& samples_ms, double& mean, double& median, double& p95, double& stddev) {
    if (samples_ms.empty()) {
        mean = median = p95 = stddev = 0.0; 
        return; 
    }
    
    double s = 0.0; 
    for (auto x : samples_ms) s += x;
    mean = s / samples_ms.size();

    double sq_sum = 0.0;
    for (auto x : samples_ms) {
        double diff = x - mean;
        sq_sum += diff * diff;
    }
    stddev = sqrt(sq_sum / samples_ms.size());

    std::vector<double> v = samples_ms;
    std::sort(v.begin(), v.end());
    median = v[v.size()/2];

    size_t idx95 = static_cast<size_t>(0.95 * (v.size()-1));
    p95 = v[idx95];
}

class Collector {
    vector<PerfData> data;
    
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
    string precision;

    void gpu_info(int method) {
        device_id = 0;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

        gpu_name = prop.name;
        cc_major = prop.major; 
        cc_minor = prop.minor;
        sm_count = prop.multiProcessorCount;
        shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
        l2_size_bytes = static_cast<long long>(prop.l2CacheSize);
        total_global_mem = static_cast<long long>(prop.totalGlobalMem);
        mem_bandwidth_GBps_est = estimate_mem_bw_GBs(prop);
        peak_flops_fp32_GFLOPs_est = peak_fp32_gflops(prop);
        max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        max_threads_per_block = prop.maxThreadsPerBlock;
        warp_size = prop.warpSize;

        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));

        algorithm = (method == 0) ? "cuBLAS" : "Custom-MM";
        precision = "fp32";

        printf("GPU: %s\n", gpu_name.c_str());
        printf("Algorithm: %s\n", algorithm.c_str());
        printf("Compute Capability: %d.%d\n", cc_major, cc_minor);
        printf("Peak FP32: %.1f GFLOPS\n", peak_flops_fp32_GFLOPs_est);
        printf("Memory Bandwidth: %.1f GB/s\n", mem_bandwidth_GBps_est);
        printf("\n");
    }

public:
    Collector(int method) {
        gpu_info(method);
    }

    PerfData benchmark(int M, int N, int K, int inner_iters = 100, int repeats = 10) {
        printf("Benchmarking: M=%d, N=%d, K=%d (%.1f GFLOPS theoretical)\n", 
               M, N, K, (2.0 * M * N * K) / 1e9);
        
        PerfData perf_data;
        perf_data.inner_iters = inner_iters;
        perf_data.repeats = repeats;

        vector<float> h_A(M * K, 1.0f);
        vector<float> h_B(K * N, 1.0f);

        GEMM gemm(M, N, K, (algorithm == "cuBLAS") ? 0 : 1);
        gemm.host_to_device(h_A.data(), h_B.data());

        // Calculate occupancy and kernel attributes
        double theoretical_occ      = 0.0;
        int    blocks_per_sm_val    = 0;

        // Launch shape for our tiled kernel
        const int block_x = TILE_WIDTH;
        const int block_y = TILE_WIDTH;
        const int grid_x  = (N + TILE_WIDTH - 1) / TILE_WIDTH;
        const int grid_y  = (M + TILE_WIDTH - 1) / TILE_WIDTH;

        int registers_per_thread   = 0;
        int shared_mem_per_block   = 0;
        const int blockSize        = TILE_WIDTH * TILE_WIDTH;

            
        cudaFuncAttributes attr{};
        if (cudaFuncGetAttributes(&attr, matrixMulKernel) == cudaSuccess) {
            registers_per_thread = attr.numRegs;
            shared_mem_per_block = attr.sharedSizeBytes;
        } else {
            printf("Warning: failed to get kernel attrs, will leave regs/shared=0\n");
        }

        int tmp_blocks = 0;
        cudaError_t occ_err =
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &tmp_blocks,
                matrixMulKernel,
                blockSize,
                /*dynamicSMemBytes=*/0);

        if (occ_err == cudaSuccess && tmp_blocks > 0) {
            blocks_per_sm_val = tmp_blocks;
            theoretical_occ   = double(blocks_per_sm_val * blockSize) / max_threads_per_sm;
            if (theoretical_occ > 1.0)
                theoretical_occ = 1.0;   // clamp
        } else {
            printf("Warning: Failed to calculate occupancy: %s\n",
                cudaGetErrorString(occ_err));
        }

        // Warm-up
        for (int i = 0; i < 10; i++) gemm.run();

        vector<double> per_run_ms;
        per_run_ms.reserve(perf_data.repeats); 
        
        for (int j = 0; j < perf_data.repeats; j++) {
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

        double t_mean_ms=0, t_median_ms=0, t_p95_ms=0, t_stddev_ms=0;
        summarize(per_run_ms, t_mean_ms, t_median_ms, t_p95_ms, t_stddev_ms);

        const double ops = 2.0 * static_cast<double>(M) * N * K;
        const double io_bytes_theoretical = ((double)M*K + (double)K*N + (double)M*N) * sizeof(float);
        const double AI = ops / io_bytes_theoretical; 

        double gflops_mean   = (ops / 1e9) / (t_mean_ms   / 1000.0);
        double gflops_median = (ops / 1e9) / (t_median_ms / 1000.0);
        double gflops_p95    = (ops / 1e9) / (t_p95_ms    / 1000.0);

        double memory_throughput_GBps = (io_bytes_theoretical / 1e9) / (t_mean_ms / 1000.0);
        double memory_efficiency_pct = (memory_throughput_GBps / mem_bandwidth_GBps_est) * 100.0;
        double compute_efficiency_pct = (gflops_mean / peak_flops_fp32_GFLOPs_est) * 100.0;

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
        perf_data.precision = precision;
        perf_data.algorithm = algorithm;
        perf_data.tile_width = TILE_WIDTH;

        perf_data.theoretical_occupancy = theoretical_occ;
        perf_data.blocks_per_sm = blocks_per_sm_val;
        perf_data.grid_x = grid_x;
        perf_data.grid_y = grid_y;
        perf_data.block_x = block_x;
        perf_data.block_y = block_y;
        perf_data.registers_per_thread = registers_per_thread;
        perf_data.shared_mem_per_block = shared_mem_per_block;

        perf_data.time_ms_mean = t_mean_ms;
        perf_data.time_ms_median = t_median_ms;
        perf_data.time_ms_p95 = t_p95_ms;
        perf_data.time_ms_stddev = t_stddev_ms;

        perf_data.gflops_mean = gflops_mean;
        perf_data.gflops_median = gflops_median;
        perf_data.gflops_p95 = gflops_p95;

        perf_data.ops_total = ops;
        perf_data.io_bytes_theoretical = io_bytes_theoretical;
        perf_data.arithmetic_intensity_ops_per_byte = AI;
        perf_data.gops_over_peak_mean = gflops_mean / peak_flops_fp32_GFLOPs_est;
        
        perf_data.memory_throughput_GBps = memory_throughput_GBps;
        perf_data.memory_efficiency_pct = memory_efficiency_pct;
        perf_data.compute_efficiency_pct = compute_efficiency_pct;

        return perf_data;
    }

    void differ_size_loop(int num_samples) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> uniform(0.0, 1.0);

        for (int i = 0; i < num_samples; i++) {
            double r1 = uniform(gen);
            double r2 = uniform(gen);
            double r3 = uniform(gen);
            
            int M = 128 + (int)(r1 * r1 * (8192 - 128));
            int N = 128 + (int)(r2 * r2 * (8192 - 128));
            int K = 128 + (int)(r3 * r3 * (8192 - 128));

            data.push_back(benchmark(M, N, K));
            cudaDeviceSynchronize();
        
            if ((i + 1) % 1000 == 0) {
                save_to_csv("benchmark_results.csv");
                data.clear(); 
            }
        }
    }

    void save_to_csv(const std::string& filename) {
        std::ifstream fin(filename);
        bool exists = fin.good();
        fin.close();

        std::ofstream f(filename, std::ios::app);
        if (!f.is_open()) return;
        if (!exists) write_csv_header(f);
        for (const auto& d : data) write_csv_row(f, d);
        f.close();
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <method> <num_samples>\n", argv[0]);
        printf("  method: 0 = cuBLAS, 1 = Custom-Tiled\n");
        printf("  num_samples: number of samples to run\n");
        return 1;
    }

    int method = std::stoi(argv[1]);
    int num_samples = std::stoi(argv[2]);

    if (method != 0 && method != 1) {
        printf("Error: method must be 0 (cuBLAS) or 1 (Custom-MM)\n");
        return 1;
    }

    if (num_samples <= 0) {
        printf("Error: num_samples must be > 0\n");
        return 1;
    }

    Collector collector(method);
    collector.differ_size_loop(num_samples);
    collector.save_to_csv("benchmark_results.csv");
    return 0;
}
