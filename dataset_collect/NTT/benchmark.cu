#include "ntt.cu"
#include <cuda_runtime.h>
#include <nvml.h>
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
    // Static GPU info
    string gpu_name;                   // GPU name
    int device_id;                     // Device ID (0)
    int cc_major, cc_minor;            // Compute Capability major.minor
    int sm_count;                      // number of streaming multiprocessors
    long long l2_size_bytes;           // L2 cache size in bytes
    int shared_mem_per_sm;             // Shared memory per SM in bytes
    long long total_global_mem;        // Total global memory in bytes
    double mem_bandwidth_GBps_est;     // Estimated memory bandwidth in GB/s
    double peak_flops_fp32_GFLOPs_est; // Estimated peak FP32 GFLOPs
    int driver_version;                // Driver version
    int cuda_runtime_version;          // CUDA runtime version
    int max_threads_per_sm;            // Max threads per SM
    int max_threads_per_block;         // Max threads per block
    int warp_size;                     // Warp size

    // Kernel Configuration
    int N;                             // NTT size
    uint32_t modulus;                  // Modulus p
    uint32_t primitive_root;           // Primitive root g
    int repeats;                       // Number of repetitions
    int inner_iters;                   // Number of inner iterations
    string algorithm;                  // Custom-NTT
    unsigned int seed;                 // Random seed for reproducibility
    
    // Runtime statistics (per-run averaged over inner_iters)
    double time_ms_mean;               // Mean time in milliseconds
    double time_ms_median;             // Median time in milliseconds
    double time_ms_p95;                // 95th percentile time in milliseconds
    double time_ms_stddev;             // Standard deviation of time in milliseconds
    double actual_clock_mhz;           // Actual GPU clock in MHz
    double actual_mem_clock_mhz;       // Actual GPU memory clock in MHz
    double temperature_c;              // GPU temperature in Celsius
    double power_watts;                // GPU power consumption in Watts

    // Derived
    double butterflies_total;          // (N/2)*log2(N)
    double modops_total;               // butterflies_total * 3
    double modops_per_sec;             // modops_total / mean_time
    double bytes_total_theoretical;    // ~8*N*log2(N) (read + write per element per stage)
    double memory_throughput_GBps;     // memory throughput in GB/s
    double memory_efficiency_pct;      // mem_bandwidth_GBps_est
};

// CSV functions
static void write_csv_header(std::ofstream& f) {
    f << "gpu_name,device_id,cc_major,cc_minor,sm_count,l2_size_bytes,shared_mem_per_sm,total_global_mem,"
      << "mem_bandwidth_GBps_est,peak_flops_fp32_GFLOPs_est,driver_version,cuda_runtime_version,"
      << "max_threads_per_sm,max_threads_per_block,warp_size,"
      << "N,modulus,primitive_root,repeats,inner_iters,algorithm,seed,"
      << "time_ms_mean,time_ms_median,time_ms_p95,time_ms_stddev,"
      << "actual_clock_mhz,actual_mem_clock_mhz,temperature_c,power_watts,"
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
      << r.seed << ","
      << std::fixed << std::setprecision(6) << r.time_ms_mean << ","
      << std::fixed << std::setprecision(6) << r.time_ms_median << ","
      << std::fixed << std::setprecision(6) << r.time_ms_p95 << ","
      << std::fixed << std::setprecision(6) << r.time_ms_stddev << ","
      << std::fixed << std::setprecision(0) << r.actual_clock_mhz << ","
      << std::fixed << std::setprecision(0) << r.actual_mem_clock_mhz << ","
      << std::fixed << std::setprecision(2) << r.temperature_c << ","
      << std::fixed << std::setprecision(2) << r.power_watts << ","
      << std::fixed << std::setprecision(0) << r.butterflies_total << ","
      << std::fixed << std::setprecision(0) << r.modops_total << ","
      << std::fixed << std::setprecision(3) << r.modops_per_sec << ","
      << std::fixed << std::setprecision(0) << r.bytes_total_theoretical << ","
      << std::fixed << std::setprecision(3) << r.memory_throughput_GBps << ","
      << std::fixed << std::setprecision(3) << r.memory_efficiency_pct
      << "\n";
}

// Utility functions
static double estimate_mem_bw_GBs(const cudaDeviceProp& p) {
  double memClockHz = static_cast<double>(p.memoryClockRate) * 1000.0;
  double busBytes   = static_cast<double>(p.memoryBusWidth) / 8.0;
  return (memClockHz * busBytes * 2.0) / 1e9; // GB/s (DDR * 2)

  // CUDA 13.0
  // int memClockKHz = 0;
  // CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate,
  // 0)); int busWidthBits = 0; CUDA_CHECK(cudaDeviceGetAttribute(&busWidthBits,
  // cudaDevAttrGlobalMemoryBusWidth, 0)); double memClockHz =
  // static_cast<double>(memClockKHz) * 1000.0; double busBytes =
  // static_cast<double>(busWidthBits) / 8.0; return (memClockHz * busBytes
  // * 2.0) / 1e9; // DDR * 2
}

static int fp32_cores_per_sm(int major, int minor) {
  const int cc = major * 10 + minor;
  // Kepler
  if (cc >= 30 && cc <= 37)
    return 192; // 3.0/3.5/3.7
  // Maxwell
  if (cc >= 50 && cc <= 53)
    return 128; // 5.0/5.2/5.3
  // Pascal
  if (cc == 60)
    return 64; // GP100
  if (cc == 61 || cc == 62)
    return 128; // GP102/104/106, TX2
  // Volta
  if (cc == 70)
    return 64;
  // Turing
  if (cc == 75)
    return 64;
  // Ampere / Ada / Hopper / Blackwell
  if (cc == 80 || cc == 86 || cc == 89 || cc == 90)
    return 128;
  return 128;
}

static double estimate_peak_fp32_GFLOPs(const cudaDeviceProp& p) {
    int cores = fp32_cores_per_sm_est(p.major, p.minor);
    double sm_hz = p.clockRate * 1000.0;
    return static_cast<double>(p.multiProcessorCount) * 
        cores * 2.0 * (sm_hz / 1e9) * 1.2; // FMA and boost

  // CUDA 13.0
  // int clockKHz = 0;
  // CUDA_CHECK(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0));
  // const double ghz = static_cast<double>(clockKHz) / 1e6;
  // return static_cast<double>(p.multiProcessorCount)
  //      * fp32_cores_per_sm(p.major, p.minor) * 2.0 * ghz * 1.2; // FMA and
  //      boost
}

static void summarize(const std::vector<double>& ms, double& mean,
                      double& median, double& p95, double& stddev) {
    if (ms.empty()) { 
        mean = median = p95 = stddev = 0.0; 
        return; 
    }
    
    // Mean
    double s = 0.0; 
    for (auto x : ms) s += x; 
    mean = s / ms.size();
    
    // Stddev
    double sq = 0.0; 
    for (auto x : ms) { 
        double d = x - mean; 
        sq += d * d; 
    } 
    stddev = sqrt(sq / ms.size());
    
    // Median and p95
    vector<double> v = ms; 
    sort(v.begin(), v.end()); 
    median = v[v.size() / 2];
    
    size_t i95 = (size_t)(0.95 * (v.size() - 1)); 
    p95 = v[i95];
}

class Collector {
    vector<PerfData> data;

    // Static GPU Specs
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
    unsigned int current_seed;

    // NVML
    nvmlDevice_t nvml_device;
    bool nvml_available;

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
        peak_flops_fp32_GFLOPs_est = estimate_peak_fp32_GFLOPs(prop);
        max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        max_threads_per_block = prop.maxThreadsPerBlock;
        warp_size = prop.warpSize;

        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));

        // Initialize NVML
        nvml_available = false;
        if (nvmlInit() == NVML_SUCCESS) {
            char pciBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
            if (cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device_id) == cudaSuccess) {
                if (nvmlDeviceGetHandleByPciBusId(pciBusId, &nvml_device) == NVML_SUCCESS) {
                    nvml_available = true;
                    printf("NVML initialized successfully\n");
                }
            }
        }
        
        if (!nvml_available) {
            printf("NVML not available - runtime metrics will be set to 0\n");
        }

        if (method == 1) {
            algorithm = "Custom-NTT";
        } else { 
            fprintf(stderr, "Invalid method=%d (use 1=Custom-NTT)\n", method); 
            std::exit(1); 
        }

        printf("GPU: %s\n", gpu_name.c_str());
        printf("Algorithm: %s\n", algorithm.c_str());
        printf("Compute Capability: %d.%d\n", cc_major, cc_minor);
    }

public:
    Collector(int method) {
        gpu_info(method);
    }

    ~Collector() {
        if (nvml_available) {
            nvmlShutdown();
        }
    }

    void collect_runtime_metrics(double& clock_mhz, double& mem_clock_mhz,
                                 double& temp_c, double& power_w) {
        // Initialize to 0 in case NVML is not available
        clock_mhz = 0.0;
        mem_clock_mhz = 0.0;
        temp_c = 0.0;
        power_w = 0.0;

        if (!nvml_available) {
            return;
        }

        // Get GPU clock (SM clock)
        unsigned int clock = 0;
        if (nvmlDeviceGetClockInfo(nvml_device, NVML_CLOCK_SM, &clock) == NVML_SUCCESS) {
            clock_mhz = static_cast<double>(clock);
        }

        // Get memory clock
        unsigned int mem_clock = 0;
        if (nvmlDeviceGetClockInfo(nvml_device, NVML_CLOCK_MEM, &mem_clock) == NVML_SUCCESS) {
            mem_clock_mhz = static_cast<double>(mem_clock);
        }

        // Get temperature
        unsigned int temp = 0;
        if (nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            temp_c = static_cast<double>(temp);
        }

        // Get power usage
        unsigned int power = 0;
        if (nvmlDeviceGetPowerUsage(nvml_device, &power) == NVML_SUCCESS) {
            power_w = static_cast<double>(power) / 1000.0; // Convert from milliwatts to watts
        }
    }


    PerfData benchmark_NTT(int N, uint32_t modulus = 998244353u, uint32_t primitive_root = 3u,
                          unsigned int seed_for_this_run = 0,
                          int inner_iters = 50, int repeats = 10) {
        printf("Benchmarking: N=%d (2^%d), modulus=%u\n", N, (int)log2(N), modulus);
        
        PerfData r;
        r.N = N; 
        r.modulus = modulus; 
        r.primitive_root = primitive_root;
        r.repeats = repeats; 
        r.inner_iters = inner_iters; 
        r.algorithm = "Custom-NTT";
        r.seed = seed_for_this_run;

        // Random input (uniform in [0, p-1])
        std::vector<uint32_t> h(N);
        {
            std::mt19937 rng(seed_for_this_run);
            std::uniform_int_distribution<uint32_t> dist(0, modulus - 1);
            for (int i = 0; i < N; ++i) {
                h[i] = dist(rng);
            }
        }

        NTT ntt(N, 1, modulus, primitive_root);
        ntt.host_to_device(h.data());

        // Warmup (no NVML interference)
        for (int i = 0; i < 10; ++i) {
            ntt.forward();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        vector<double> per_run_ms; 
        per_run_ms.reserve(repeats);

        // Timed runs
        for (int rep = 0; rep < repeats; ++rep) {
            cudaEvent_t start, stop; 
            cudaEventCreate(&start); 
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            for (int it = 0; it < inner_iters; ++it) {
                ntt.forward();
            }

            cudaEventRecord(stop); 
            cudaEventSynchronize(stop);

            float total_ms = 0.f; 
            cudaEventElapsedTime(&total_ms, start, stop);
            cudaEventDestroy(start); 
            cudaEventDestroy(stop);
            per_run_ms.push_back(double(total_ms) / inner_iters);
        }

        // Collect runtime metrics AFTER all timing is complete
        // Run a few iterations to keep GPU loaded for representative measurements
        double actual_clock = 0.0, actual_mem_clock = 0.0;
        double actual_temp = 0.0, actual_power = 0.0;

        for (int i = 0; i < 5; i++) {
            ntt.forward();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        collect_runtime_metrics(actual_clock, actual_mem_clock, actual_temp, actual_power);

        // Calculate statistics
        summarize(per_run_ms, r.time_ms_mean, r.time_ms_median, r.time_ms_p95, r.time_ms_stddev);

        // Estimate metrics
        const int logN = (int)std::log2((double)N);
        const double butterflies = (double)N / 2.0 * (double)logN;        
        const double modops = butterflies * 3.0; // 1 mul_mod + 1 add_mod + 1 sub_mod per butterfly
        const double bytes_theory = 8.0 * (double)N * (double)logN; // read + write per element per stage

        r.butterflies_total = butterflies;
        r.modops_total = modops;
        r.modops_per_sec = (r.time_ms_mean > 0) ? (modops / (r.time_ms_mean / 1000.0)) : 0.0;

        r.bytes_total_theoretical = bytes_theory;
        r.memory_throughput_GBps = (r.time_ms_mean > 0) ? ((bytes_theory / 1e9) / (r.time_ms_mean / 1000.0)) : 0.0;
        r.memory_efficiency_pct = (mem_bandwidth_GBps_est > 0) ? (r.memory_throughput_GBps / mem_bandwidth_GBps_est * 100.0) : 0.0;

        r.actual_clock_mhz = actual_clock;
        r.actual_mem_clock_mhz = actual_mem_clock;
        r.temperature_c = actual_temp;
        r.power_watts = actual_power;

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

    std::string differ_size_loop(int num_samples, const std::string& output_name = "", unsigned int seed = 0) {
        // If seed is 0, use random seed; otherwise use provided seed
        unsigned int actual_seed = (seed == 0) ? random_device{}() : seed;
        current_seed = actual_seed;

        // Build filename
        std::string filename = build_filename(output_name, num_samples, actual_seed);

        mt19937 gen(actual_seed);
        uniform_real_distribution<> U(0.0, 1.0);

        printf("Starting NTT benchmarking with %d samples...\n", num_samples);
        printf("Random seed: %u %s\n", actual_seed,
               (seed == 0) ? "(randomly generated)" : "(user-provided)");
        printf("Output file: %s\n\n", filename.c_str());

        for (int i = 0; i < num_samples; ++i) {
            int k = 15 + (int)std::floor(U(gen) * (24 - 15 + 1));  // k ∈ [15,24] uniform
            int N = 1 << k;

            data.push_back(benchmark_NTT(N, 998244353u, 3u, actual_seed));
            CUDA_CHECK(cudaDeviceSynchronize());

            if ((i + 1) % 100 == 0) {
                save_to_csv(filename);
                data.clear();
                printf("Checkpoint saved (%d samples so far)\n\n", i + 1);
            }
        }
        printf("Done!\n");
        return filename; // Return filename for final save
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

    // Helper function to build filename
    static std::string build_filename(const std::string& output_name, int num_samples, unsigned int seed) {
        std::string filename = "benchmark_results";
        if (!output_name.empty()) {
            filename += "_" + output_name;
        }
        filename += "_N" + std::to_string(num_samples);
        if (seed != 0) {
            filename += "_S" + std::to_string(seed);
        }
        filename += ".csv";
        return filename;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <method> <num_samples> [output_name] [N] [repeats] [inner_iters] [seed]\n", argv[0]);
        printf("  method: 1 = Custom-NTT\n");
        printf("  num_samples: number of samples to run\n");
        printf("  output_name: (optional) custom name for output file\n");
        printf("  N: (optional) fixed NTT size (power of 2). If not specified, random sizes will be used\n");
        printf("  repeats: (optional) number of repetitions (default: 10)\n");
        printf("  inner_iters: (optional) inner iterations (default: 50)\n");
        printf("  seed: (optional) random seed for reproducibility (default: random)\n");
        printf("\nOutput file naming: benchmark_results_[output_name_]N{num_samples}[_S{seed}].csv\n");
        printf("\nExamples:\n");
        printf("  %s 1 1000                      # Output: benchmark_results_N1000.csv\n", argv[0]);
        printf("  %s 1 1000 a100_ntt             # Output: benchmark_results_a100_ntt_N1000.csv\n", argv[0]);
        printf("  %s 1 500 test 65536 10 50 42   # Output: benchmark_results_test_N500_S42.csv (fixed N=65536)\n", argv[0]);
        printf("  %s 1 1000 42                   # Output: benchmark_results_N1000_S42.csv (seed only)\n", argv[0]);
        return 1;
    }

    int method = std::stoi(argv[1]);
    int num_samples = std::stoi(argv[2]);
    std::string output_name = "";
    int N = -1;
    int repeats = 10;
    int inner_iters = 50;
    unsigned int seed = 0; // 0 means random seed

    // Parse optional arguments - need to determine if argv[3] is output_name or N (or seed)
    if (argc >= 4) {
        std::string arg3 = argv[3];
        bool is_number = true;
        for (char c : arg3) {
            if (!isdigit(c)) {
                is_number = false;
                break;
            }
        }

        if (is_number) {
            // argv[3] is either N or seed
            int value = std::stoi(argv[3]);
            // If value is a power of 2 >= 2, treat as N; otherwise treat as seed
            if (is_power_of_two(value) && value >= 2) {
                N = value;
                if (argc >= 5) repeats = std::stoi(argv[4]);
                if (argc >= 6) inner_iters = std::stoi(argv[5]);
                if (argc >= 7) seed = static_cast<unsigned int>(std::stoul(argv[6]));
            } else {
                // Treat as seed
                seed = static_cast<unsigned int>(value);
            }
        } else {
            // argv[3] is output_name
            output_name = argv[3];

            // Parse remaining arguments
            if (argc >= 5) {
                int value = std::stoi(argv[4]);
                if (is_power_of_two(value) && value >= 2) {
                    N = value;
                    if (argc >= 6) repeats = std::stoi(argv[5]);
                    if (argc >= 7) inner_iters = std::stoi(argv[6]);
                    if (argc >= 8) seed = static_cast<unsigned int>(std::stoul(argv[7]));
                } else {
                    // Treat as seed
                    seed = static_cast<unsigned int>(value);
                }
            }
        }
    }

    if (method != 1) {
        printf("Error: method must be 1 (Custom-NTT)\n");
        return 1;
    }
    if (num_samples <= 0) {
        printf("Error: num_samples must be > 0\n");
        return 1;
    }

    Collector collector(method);

    // Generate actual seed for filename consistency
    unsigned int actual_seed = (seed == 0) ? random_device{}() : seed;
    std::string filename = Collector::build_filename(output_name, num_samples, actual_seed);

    if (N > 0) {
        // Fixed N mode
        if (!is_power_of_two(N) || N < 2) {
            printf("Error: N must be a power of 2 and >= 2\n");
            return 1;
        }

        printf("Running fixed-N benchmarks (N=%d) with %d samples\n", N, num_samples);
        printf("Random seed: %u %s\n", actual_seed,
               (seed == 0) ? "(randomly generated)" : "(user-provided)");
        printf("Output file: %s\n\n", filename.c_str());

        for (int i = 0; i < num_samples; ++i) {
            auto rec = collector.benchmark_NTT(N, 998244353u, 3u, actual_seed, inner_iters, repeats);

            std::ifstream fin(filename);
            bool exists = fin.good();
            fin.close();

            std::ofstream f(filename, std::ios::app);
            if (!f.is_open()) {
                printf("Failed to open output\n");
                return 1;
            }
            if (!exists) write_csv_header(f);
            write_csv_row(f, rec);
            f.close();

            if ((i + 1) % 100 == 0) {
                printf("Completed %d samples\n", i + 1);
            }
        }
        printf("Done fixed-N runs.\n");
    }
    else {
        // Random N mode - filename is built and returned by differ_size_loop
        filename = collector.differ_size_loop(num_samples, output_name, seed);
        collector.save_to_csv(filename);
    }

    return 0;
}
