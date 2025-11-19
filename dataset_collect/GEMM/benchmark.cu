#include "gemm.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <nvml.h>
#include <random>
#include <string>
#include <vector>

using namespace std;

struct PerfData {
  // Static GPU info
  string gpu_name;                   // GPU name
  int device_id;                     // Device ID (0)
  int cc_major, cc_minor;            // Compute capability major.minor
  int sm_count;                      // Number of streaming multiprocessors
  long long l2_size_bytes;           // L2 cache size in bytes
  int shared_mem_per_sm;             // Shared memory per SM in bytes
  long long total_global_mem;        // Total global memory in bytes
  double mem_bandwidth_GBps_est;     // Estimated memory bandwidth in GB/s
  double peak_flops_fp32_GFLOPs_est; // Estimated peak FLOPS for FP32 in GFLOPS
  int driver_version;                // Driver version
  int cuda_runtime_version;          // CUDA runtime version
  int max_threads_per_sm;            // Max threads per SM
  int max_threads_per_block;         // Max threads per block
  int warp_size;                     // Warp size

  // Kernel configuration
  int M, N, K;                       // Matrix dimensions
  string precision;                  // FP32
  int repeats;                       // Number of repetitions
  int inner_iters;                   // Number of inner iterations
  string algorithm;                  // cuBLAS or Custom-MM
  unsigned int seed;                 // Random seed for reproducibility

  int tile_width;                    // Tile width for custom kernel
  double theoretical_occupancy;      // Theoretical occupancy
  int blocks_per_sm;                 // Blocks per SM
  int grid_x, grid_y;                // Grid dimensions
  int block_x, block_y;              // Block dimensions
  int registers_per_thread;          // Registers per thread
  int shared_mem_per_block;          // Shared memory per block

  // Runtime statistics (per-run averaged over inner_iters)
  double time_ms_mean;               // Mean time in milliseconds
  double time_ms_median;             // Median time in milliseconds
  double time_ms_p95;                // 95th percentile time in milliseconds
  double time_ms_stddev;             // Standard deviation of time in milliseconds
  double gflops_mean;                // Mean GFLOPS
  double gflops_median;              // Median GFLOPS
  double gflops_p95;                 // 95th percentile GFLOPS
  double actual_clock_mhz;           // actual GPU clock in MHz
  double actual_mem_clock_mhz;       // actual GPU memory clock in MHz
  double temperature_c;              // GPU temperature in Celsius
  double power_watts;                // GPU power consumption in Watts

  // Derived
  double ops_total;                  // total operations (2*M*N*K)
  double io_bytes_theoretical;       // theoretical IO bytes (MK+KN+MN)*sizeof(float)
  double gops_over_peak_mean;        // gflops_mean / peak_flops_fp32_GFLOPs_est
  double arithmetic_intensity_ops_per_byte;
  // arithmetic intensity (ops / io_bytes_theoretical)
  double memory_throughput_GBps;     // memory throughput in GB/s
  double memory_efficiency_pct;      // memory efficiency in %
  double compute_efficiency_pct;     // compute efficiency in % (vs theoretical peak)

  // Actual clock-based metrics
  double peak_flops_fp32_GFLOPs_actual; // peak FLOPS based on actual clock
  double compute_efficiency_actual_pct; // compute efficiency vs actual peak
};

static void write_csv_header(std::ofstream &f) {
  f << "gpu_name,device_id,cc_major,cc_minor,sm_count,l2_size_bytes,shared_mem_"
       "per_sm,total_global_mem,"
    << "mem_bandwidth_GBps_est,peak_flops_fp32_GFLOPs_est,driver_version,cuda_"
       "runtime_version,"
    << "max_threads_per_sm,max_threads_per_block,warp_size,"
    << "M,N,K,precision,repeats,inner_iters,algorithm,seed,tile_width,"
    << "theoretical_occupancy,blocks_per_sm,grid_x,grid_y,block_x,block_y,"
    << "registers_per_thread,shared_mem_per_block,"
    << "time_ms_mean,time_ms_median,time_ms_p95,time_ms_stddev,"
    << "gflops_mean,gflops_median,gflops_p95,"
    << "actual_clock_mhz,actual_mem_clock_mhz,temperature_c,power_watts,"
    << "ops_total,io_bytes_theoretical,gops_over_peak_mean,arithmetic_"
       "intensity_ops_per_byte,"
    << "memory_throughput_GBps,memory_efficiency_pct,compute_efficiency_pct,"
    << "peak_flops_fp32_GFLOPs_actual,compute_efficiency_actual_pct\n";
}

static void write_csv_row(std::ofstream &f, const PerfData &r) {
  f << r.gpu_name << "," << r.device_id << "," << r.cc_major << ","
    << r.cc_minor << "," << r.sm_count << "," << r.l2_size_bytes << ","
    << r.shared_mem_per_sm << "," << r.total_global_mem << "," << std::fixed
    << std::setprecision(3) << r.mem_bandwidth_GBps_est << "," << std::fixed
    << std::setprecision(3) << r.peak_flops_fp32_GFLOPs_est << ","
    << r.driver_version << "," << r.cuda_runtime_version << ","
    << r.max_threads_per_sm << "," << r.max_threads_per_block << ","
    << r.warp_size << "," << r.M << "," << r.N << "," << r.K << ","
    << r.precision << "," << r.repeats << "," << r.inner_iters << ","
    << r.algorithm << "," << r.seed << "," << r.tile_width << "," << std::fixed
    << std::setprecision(6) << r.theoretical_occupancy << "," << r.blocks_per_sm
    << "," << r.grid_x << "," << r.grid_y << "," << r.block_x << ","
    << r.block_y << "," << r.registers_per_thread << ","
    << r.shared_mem_per_block << "," << std::fixed << std::setprecision(6)
    << r.time_ms_mean << "," << std::fixed << std::setprecision(6)
    << r.time_ms_median << "," << std::fixed << std::setprecision(6)
    << r.time_ms_p95 << "," << std::fixed << std::setprecision(6)
    << r.time_ms_stddev << "," << std::fixed << std::setprecision(3)
    << r.gflops_mean << "," << std::fixed << std::setprecision(3)
    << r.gflops_median << "," << std::fixed << std::setprecision(3)
    << r.gflops_p95 << "," << std::fixed << std::setprecision(0)
    << r.actual_clock_mhz << "," << std::fixed << std::setprecision(0)
    << r.actual_mem_clock_mhz << "," << std::fixed << std::setprecision(2)
    << r.temperature_c << "," << std::fixed << std::setprecision(2)
    << r.power_watts << "," << std::fixed << std::setprecision(0) << r.ops_total
    << "," << std::fixed << std::setprecision(0) << r.io_bytes_theoretical
    << "," << std::fixed << std::setprecision(6) << r.gops_over_peak_mean << ","
    << std::fixed << std::setprecision(6) << r.arithmetic_intensity_ops_per_byte
    << "," << std::fixed << std::setprecision(3) << r.memory_throughput_GBps
    << "," << std::fixed << std::setprecision(3) << r.memory_efficiency_pct
    << "," << std::fixed << std::setprecision(3) << r.compute_efficiency_pct
    << "," << std::fixed << std::setprecision(3)
    << r.peak_flops_fp32_GFLOPs_actual << "," << std::fixed
    << std::setprecision(3) << r.compute_efficiency_actual_pct << "\n";
}

// Utility functions
static double estimate_mem_bw_GBs(const cudaDeviceProp &p) {
  double memClockHz = static_cast<double>(p.memoryClockRate) * 1000.0;
  double busBytes = static_cast<double>(p.memoryBusWidth) / 8.0;
  return (memClockHz * busBytes * 2.0) / 1e9; // DDR * 2

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

static double peak_fp32_gflops(const cudaDeviceProp &p) {
  const double ghz = static_cast<double>(p.clockRate) / 1e6;
  return static_cast<double>(p.multiProcessorCount) *
         fp32_cores_per_sm(p.major, p.minor) * 2.0 * ghz * 1.2; // FMA and boost

  // CUDA 13.0
  // int clockKHz = 0;
  // CUDA_CHECK(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0));
  // const double ghz = static_cast<double>(clockKHz) / 1e6;
  // return static_cast<double>(p.multiProcessorCount)
  //      * fp32_cores_per_sm(p.major, p.minor) * 2.0 * ghz * 1.2; // FMA and
  //      boost
}

static void summarize(const std::vector<double> &ms, double &mean,
                      double &median, double &p95, double &stddev) {
  if (ms.empty()) {
    mean = median = p95 = stddev = 0.0;
    return;
  }

  // Mean
  double s = 0.0;
  for (auto x : ms)
    s += x;
  mean = s / ms.size();

  // Stddev
  double sq_sum = 0.0;
  for (auto x : ms) {
    double diff = x - mean;
    sq_sum += diff * diff;
  }
  stddev = sqrt(sq_sum / ms.size());

  // Median and P95
  std::vector<double> v = ms;
  std::sort(v.begin(), v.end());
  median = v[v.size() / 2];

  size_t idx95 = static_cast<size_t>(0.95 * (v.size() - 1));
  p95 = v[idx95];
}

class Collector {
  vector<PerfData> data;

  // Static GPU info
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
    peak_flops_fp32_GFLOPs_est = peak_fp32_gflops(prop);
    max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    max_threads_per_block = prop.maxThreadsPerBlock;
    warp_size = prop.warpSize;
    precision = "fp32";

    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));

    // Initialize NVML (if available)
    nvml_available = false;
    if (nvmlInit() == NVML_SUCCESS) {
      char pciBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      if (cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device_id) ==
          cudaSuccess) {
        if (nvmlDeviceGetHandleByPciBusId(pciBusId, &nvml_device) ==
            NVML_SUCCESS) {
          nvml_available = true;
          printf("NVML initialized successfully\n");
        }
      }
    }

    if (!nvml_available) {
      printf("NVML not available - runtime metrics will be set to 0\n");
    }

    if (method == 0) {
      algorithm = "cuBLAS";
    } else if (method == 1) {
      algorithm = "Custom-MM";
    } else {
      fprintf(stderr, "Invalid method=%d (use 0=cublas, 1=custom)\n", method);
      std::exit(1);
    }

    printf("GPU: %s\n", gpu_name.c_str());
    printf("Algorithm: %s\n", algorithm.c_str());
    printf("Compute Capability: %d.%d\n", cc_major, cc_minor);
    printf("\n");
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

  void collect_runtime_metrics(double &clock_mhz, double &mem_clock_mhz,
                               double &temp_c, double &power_w) {
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
      power_w = static_cast<double>(power) / 1000.0; // Milliwatts to watts
    }
  }

  PerfData benchmark(int M, int N, int K, unsigned int seed_for_this_run,
                     int inner_iters = 100, int repeats = 10) {
    printf("Benchmarking: M=%d, N=%d, K=%d (%.1f GFLOPS theoretical)\n", M, N,
           K, (2.0 * M * N * K) / 1e9);

    PerfData r;
    r.inner_iters = inner_iters;
    r.repeats = repeats;
    r.seed = seed_for_this_run;

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);

    GEMM gemm(M, N, K, (algorithm == "cuBLAS") ? 0 : 1);
    gemm.host_to_device(h_A.data(), h_B.data());

    // Calculate occupancy and kernel attributes
    double theoretical_occ = 0.0;
    int blocks_per_sm_val = 0;

    // Launch shape for our tiled kernel
    const int block_x = TILE_WIDTH;
    const int block_y = TILE_WIDTH;
    const int grid_x = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    const int grid_y = (M + TILE_WIDTH - 1) / TILE_WIDTH;

    int registers_per_thread = 0;
    int shared_mem_per_block = 0;
    const int blockSize = TILE_WIDTH * TILE_WIDTH;

    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, matrixMulKernel) == cudaSuccess) {
      registers_per_thread = attr.numRegs;
      shared_mem_per_block = attr.sharedSizeBytes;
    } else {
      printf("Warning: failed to get kernel attrs, will leave regs/shared=0\n");
    }

    int tmp_blocks = 0;
    cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &tmp_blocks, matrixMulKernel, blockSize,
        /*dynamicSMemBytes=*/0);

    if (occ_err == cudaSuccess && tmp_blocks > 0) {
      blocks_per_sm_val = tmp_blocks;
      theoretical_occ =
          double(blocks_per_sm_val * blockSize) / max_threads_per_sm;
      if (theoretical_occ > 1.0)
        theoretical_occ = 1.0; // clamp
    } else {
      printf("Warning: Failed to calculate occupancy: %s\n",
             cudaGetErrorString(occ_err));
    }

    // Warm-up (no NVML interference)
    for (int i = 0; i < 10; i++)
      gemm.run();
    cudaDeviceSynchronize();

    std::vector<double> per_run_ms;
    per_run_ms.reserve(r.repeats);

    // Timed runs
    for (int j = 0; j < r.repeats; j++) {
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      for (int i = 0; i < r.inner_iters; i++) {
        gemm.run();
      }
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      float per_total_time = 0.f;
      cudaEventElapsedTime(&per_total_time, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      double avg_ms =
          static_cast<double>(per_total_time) / r.inner_iters;
      per_run_ms.push_back(avg_ms);
    }

    // Collect runtime metrics AFTER all timing is complete
    // Run a few iterations to keep GPU loaded for representative measurements
    double actual_clock = 0.0, actual_mem_clock = 0.0;
    double actual_temp = 0.0, actual_power = 0.0;

    for (int i = 0; i < 5; i++) {
      gemm.run();
    }
    cudaDeviceSynchronize();

    collect_runtime_metrics(actual_clock, actual_mem_clock, actual_temp,
                            actual_power);

    // Calculate statistics
    double t_mean_ms = 0, t_median_ms = 0, t_p95_ms = 0, t_stddev_ms = 0;
    summarize(per_run_ms, t_mean_ms, t_median_ms, t_p95_ms, t_stddev_ms);

    // Calculate metrics
    const double ops = 2.0 * static_cast<double>(M) * N * K;
    const double io_bytes_theoretical =
        ((double)M * K + (double)K * N + (double)M * N) * sizeof(float);
    const double AI = ops / io_bytes_theoretical;

    double gflops_mean = (ops / 1e9) / (t_mean_ms / 1000.0);
    double gflops_median = (ops / 1e9) / (t_median_ms / 1000.0);
    double gflops_p95 = (ops / 1e9) / (t_p95_ms / 1000.0);

    double memory_throughput_GBps =
        (io_bytes_theoretical / 1e9) / (t_mean_ms / 1000.0);
    double memory_efficiency_pct =
        (memory_throughput_GBps / mem_bandwidth_GBps_est) * 100.0;
    double compute_efficiency_pct =
        (gflops_mean / peak_flops_fp32_GFLOPs_est) * 100.0;

    // Fill PerfData structure
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
    r.M = M;
    r.N = N;
    r.K = K;
    r.precision = precision;
    r.algorithm = algorithm;
    r.tile_width = TILE_WIDTH;

    r.theoretical_occupancy = theoretical_occ;
    r.blocks_per_sm = blocks_per_sm_val;
    r.grid_x = grid_x;
    r.grid_y = grid_y;
    r.block_x = block_x;
    r.block_y = block_y;
    r.registers_per_thread = registers_per_thread;
    r.shared_mem_per_block = shared_mem_per_block;
    r.time_ms_mean = t_mean_ms;
    r.time_ms_median = t_median_ms;
    r.time_ms_p95 = t_p95_ms;
    r.time_ms_stddev = t_stddev_ms;

    r.gflops_mean = gflops_mean;
    r.gflops_median = gflops_median;
    r.gflops_p95 = gflops_p95;

    r.actual_clock_mhz = actual_clock;
    r.actual_mem_clock_mhz = actual_mem_clock;
    r.temperature_c = actual_temp;
    r.power_watts = actual_power;

    r.ops_total = ops;
    r.io_bytes_theoretical = io_bytes_theoretical;
    r.arithmetic_intensity_ops_per_byte = AI;
    r.gops_over_peak_mean =
        (peak_flops_fp32_GFLOPs_est > 0.0)
            ? (gflops_mean / peak_flops_fp32_GFLOPs_est)
            : 0.0;

    r.memory_throughput_GBps = memory_throughput_GBps;
    r.memory_efficiency_pct = memory_efficiency_pct;
    r.compute_efficiency_pct = compute_efficiency_pct;

    // If we have actual clock measurements, recalculate efficiency based on
    // actual clock
    if (actual_clock > 0.0) {
      // Calculate actual peak FLOPS at measured clock speed
      double actual_peak_gflops = static_cast<double>(sm_count) *
                                  fp32_cores_per_sm(cc_major, cc_minor) * 2.0 *
                                  (actual_clock / 1000.0); // GHz

      // Recalculate compute efficiency based on actual measured clock
      double compute_efficiency_actual =
          (gflops_mean / actual_peak_gflops) * 100.0;
    }

    // Calculate actual peak FLOPS based on measured clock speed
    if (actual_clock > 0.0) {
      const double actual_ghz = actual_clock / 1000.0; // MHz to GHz
      r.peak_flops_fp32_GFLOPs_actual =
          static_cast<double>(sm_count) *
          fp32_cores_per_sm(cc_major, cc_minor) * 2.0 *
          actual_ghz; // FMA, no boost multiplier (actual clock already includes
                      // boost)

      r.compute_efficiency_actual_pct =
          (gflops_mean / r.peak_flops_fp32_GFLOPs_actual) * 100.0;
    } else {
      r.peak_flops_fp32_GFLOPs_actual = 0.0;
      r.compute_efficiency_actual_pct = 0.0;
    }

    return r;
  }

  std::string differ_size_loop(int num_samples, const std::string& output_name = "", unsigned int seed = 0) {
    // If seed is 0, use random seed; otherwise use provided seed
    unsigned int actual_seed = (seed == 0) ? random_device{}() : seed;
    current_seed = actual_seed; // Store for reference

    // Build filename
    std::string filename = build_filename(output_name, num_samples, actual_seed);

    mt19937 gen(actual_seed);
    uniform_real_distribution<> uniform(0.0, 1.0);

    printf("Starting benchmarking with %d samples...\n", num_samples);
    printf("Random seed: %u %s\n", actual_seed,
           (seed == 0) ? "(randomly generated)" : "(user-provided)");
    printf("Output file: %s\n\n", filename.c_str());

    for (int i = 0; i < num_samples; i++) {
      double r1 = uniform(gen);
      double r2 = uniform(gen);
      double r3 = uniform(gen);

      int M = 128 + (int)(r1 * r1 * (8192 - 128));
      int N = 128 + (int)(r2 * r2 * (8192 - 128));
      int K = 128 + (int)(r3 * r3 * (8192 - 128));

      data.push_back(benchmark(M, N, K, actual_seed));
      cudaDeviceSynchronize();

      // Periodic save for every 100 samples
      if ((i + 1) % 100 == 0) {
        save_to_csv(filename);
        data.clear();
        printf("Checkpoint saved (%d samples so far)\n\n", i + 1);
      }
    }
    printf("Done!\n");
    return filename; // Return filename for final save
  }

  void save_to_csv(const std::string &filename) {
    std::ifstream fin(filename);
    bool exists = fin.good();
    fin.close();

    std::ofstream f(filename, std::ios::app);
    if (!f.is_open()) {
      printf("Failed to open file: %s\n", filename.c_str());
      return;
    }
    if (!exists)
      write_csv_header(f);
    for (const auto &d : data)
      write_csv_row(f, d);
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

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <method> <num_samples> [output_name] [seed]\n", argv[0]);
    printf("  method: 0 = cuBLAS, 1 = Custom-MM\n");
    printf("  num_samples: number of samples to run\n");
    printf("  output_name: (optional) custom name for output file\n");
    printf("  seed: (optional) random seed for reproducibility (default: random)\n");
    printf("\nOutput file naming: benchmark_results_[output_name_]N{num_samples}[_S{seed}].csv\n");
    printf("\nExamples:\n");
    printf("  %s 0 1000                    # Output: benchmark_results_N1000.csv\n", argv[0]);
    printf("  %s 0 1000 rtx4090            # Output: benchmark_results_rtx4090_N1000.csv\n", argv[0]);
    printf("  %s 0 1000 rtx4090 42         # Output: benchmark_results_rtx4090_N1000_S42.csv\n", argv[0]);
    return 1;
  }

  int method = std::stoi(argv[1]);
  int num_samples = std::stoi(argv[2]);
  std::string output_name = "";
  unsigned int seed = 0; // 0 means random seed

  // Parse optional arguments
  if (argc >= 4) {
    // Check if argv[3] is a number (seed) or string (output_name)
    bool is_number = true;
    std::string arg3 = argv[3];
    for (char c : arg3) {
      if (!isdigit(c)) {
        is_number = false;
        break;
      }
    }

    if (is_number) {
      // argv[3] is seed
      seed = static_cast<unsigned int>(std::stoul(argv[3]));
    } else {
      // argv[3] is output_name
      output_name = argv[3];

      // Check for seed in argv[4]
      if (argc >= 5) {
        seed = static_cast<unsigned int>(std::stoul(argv[4]));
      }
    }
  }

  if (method != 0 && method != 1) {
    printf("Error: method must be 0 (cuBLAS) or 1 (Custom-MM)\n");
    return 1;
  }

  if (num_samples <= 0) {
    printf("Error: num_samples must be > 0\n");
    return 1;
  }

  Collector collector(method);
  std::string filename = collector.differ_size_loop(num_samples, output_name, seed);
  collector.save_to_csv(filename);
  return 0;
}
