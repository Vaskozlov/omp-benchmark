#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdio>
#include <execution>
#include <span>
#include <fmt/format.h>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>
#include <isl/thread/async_task.hpp>
#include <isl/thread/pool.hpp>

#if defined(__x86_64__)
#    include <immintrin.h>
#else
#    include <arm_neon.h>
#endif

const static std::size_t ThreadsCount = std::thread::hardware_concurrency() / 2;

static isl::thread::Pool pool{0};

std::vector<float> generateRandomVectorOfFloats(
  const std::size_t n, const float lower_bound, const float upper_bound) {
  std::vector<float> result(n);
  std::mt19937_64 engine;
  std::uniform_real_distribution<float> distribution{lower_bound, upper_bound};

  for (auto&elem: result) {
    elem = distribution(engine);
  }

  return result;
}

std::vector<float> generateRandomVectorOfFloatsNormalDistribution(
  const std::size_t n, const float mean, const float der) {
  std::vector<float> result(n);
  std::mt19937_64 engine;
  std::normal_distribution<float> distribution{mean, der};

  for (auto&elem: result) {
    elem = distribution(engine);
  }

  return result;
}

static constexpr std::size_t dataSize = 10'000'000;

const auto randomData = generateRandomVectorOfFloats(dataSize, -1000.0F, 1000.0F);

const auto randomNormalData =
    generateRandomVectorOfFloatsNormalDistribution(dataSize, 0.0F, 1000.0F);

const auto&data = randomNormalData;

auto ompWrongParallelMax(std::span<const float> vec) -> decltype(vec.begin()) {
  std::size_t idx = 0;

  float maxValue = -std::numeric_limits<float>::infinity();
  const auto end = static_cast<std::intptr_t>(vec.size());

#pragma omp parallel for
  for (std::intptr_t i = 0; i < end; ++i) {
    if (maxValue < vec[i]) {
      maxValue = vec[i];
      idx = i;
    }
  }

  return vec.begin() + static_cast<std::ptrdiff_t>(idx);
}

auto ompParallelMax(std::span<const float> vec) -> decltype(vec.begin()) {
  float maxValue = -std::numeric_limits<float>::infinity();
  std::size_t idx = 0;

#pragma omp parallel default(none) shared(vec, maxValue, idx) num_threads(ThreadsCount)
  {
    auto localMax = maxValue;
    auto localIdx = idx;
    const auto end = static_cast<std::intptr_t>(vec.size());

#pragma omp for nowait
    for (std::intptr_t i = 0; i < end; ++i) {
      if (localMax < vec[i]) {
        localMax = vec[i];
        localIdx = i;
      }
    }

#pragma omp critical
    if (localMax > maxValue) {
      maxValue = localMax;
      idx = localIdx;
    }
  }

  return vec.begin() + static_cast<std::ptrdiff_t>(idx);
}

#ifdef __x86_64__
auto simdMax(const float *data, std::size_t n) -> const float *
{
    __m256 maxVector = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    std::size_t dataOffset = 0;
    std::size_t maxValueIdx = 0;

    for (std::size_t i = 0; i < n / 8; ++i, dataOffset += 8) {
        __m256 localData = _mm256_loadu_ps(data + dataOffset);
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(localData, maxVector, _CMP_GT_OQ));

        if (mask == 0) {
            continue;
        }

        auto maxValue = _mm_cvtss_f32(_mm256_castps256_ps128(maxVector));

        for (std::size_t j = 0; j < 8; ++j) {
            if (data[dataOffset + j] > maxValue) {
                maxValue = data[dataOffset + j];
                maxValueIdx = dataOffset + j;
            }
        }

        maxVector = _mm256_set1_ps(maxValue);
    }

    auto maxValue = _mm_cvtss_f32(_mm256_castps256_ps128(maxVector));

    while (dataOffset < n) {
        if (data[dataOffset] > maxValue) {
            maxValue = data[dataOffset];
            maxValueIdx = dataOffset;
        }
        ++dataOffset;
    }

    return data + maxValueIdx;
}
#else

auto simdMax(const float* data, const std::size_t n) -> const float* {
  std::size_t data_offset = 0;
  std::size_t max_elem_index = 0;
  float32x4_t max = vdupq_n_f32(-std::numeric_limits<float>::infinity());

  for (std::size_t i = 0; i < n / 4; ++i, data_offset += 4) {
    float32x4_t v = vld1q_f32(data + data_offset);

    auto cmp_result = vcgtq_f32(v, max);
    auto max_cmp = vmaxvq_u32(cmp_result);

    if (max_cmp == 0) {
      continue;
    }

    float max_elem = vgetq_lane_f32(max, 0);

    for (std::size_t j = 0; j < 4; ++j) {
      if (data[data_offset + j] > max_elem) {
        max_elem = data[data_offset + j];
        max_elem_index = data_offset + j;
      }
    }

    max = vdupq_n_f32(max_elem);
  }

  float max_elem = vgetq_lane_f32(max, 0);

  while (data_offset < n) {
    if (data[data_offset] > max_elem) {
      max_elem_index = data_offset;
      max_elem = data[max_elem_index];
    }
    ++data_offset;
  }

  return data + max_elem_index;
}
#endif

auto ompParallelMaxWithSimd(std::span<const float> vec) -> const float* {
  const auto* it = vec.data();

#pragma omp parallel default(none) shared(vec, it) num_threads(ThreadsCount)
  {
    const auto thread_id = static_cast<std::size_t>(omp_get_thread_num());
    const auto total_threads = static_cast<std::size_t>(omp_get_num_threads());

    // Compute the range for each thread
    std::size_t chunk_size = vec.size() / total_threads; // Equal chunk size
    std::size_t start = thread_id * chunk_size;
    std::size_t end = (thread_id == total_threads - 1) ? vec.size() : start + chunk_size;

    auto local_max = simdMax(vec.data() + start, end - start);

#pragma omp critical
    if (*local_max > *it) {
      it = local_max;
    }
  }

  return it;
}

auto trivialMax(std::span<const float> vec) -> decltype(vec.begin()) {
  float maxValue = -std::numeric_limits<float>::infinity();
  std::size_t idx = 0;

  for (std::size_t i = 0; i < randomData.size(); ++i) {
    if (maxValue < randomData[i]) {
      idx = i;
      maxValue = randomData[i];
    }
  }

  return vec.begin() + static_cast<std::ptrdiff_t>(idx);
}

auto standardMax(std::span<const float> vec) -> decltype(vec.begin()) {
  return std::ranges::max_element(vec);
}

#if !defined(_LIBCPP_VERSION)
auto standardMaxParallel(std::span<const float> vec) -> decltype(vec.begin())
{
    return std::max_element(std::execution::par, vec.begin(), vec.end());
}
#endif

auto accumulateOmpSimd(std::span<const float> vec) -> float {
  float result = 0.0F;

#pragma omp simd reduction(+ : result)
  for (std::size_t i = 0; i < vec.size(); ++i) {
    result += vec[i];
  }

  return result;
}

#if defined(__x86_64__)
auto accumulateCustomSimd(const float *data, const std::size_t n) -> float
{
    if (n < 8) {
        return std::accumulate(data, data + n, 0.0F);
    }

    __m256 accumulator = _mm256_loadu_ps(data);
    std::size_t data_offset = 8;

    for (std::size_t i = 1; i < n / 8; ++i, data_offset += 8) {
        auto tmp = _mm256_loadu_ps(data + data_offset);
        accumulator = _mm256_add_ps(accumulator, tmp);
    }

    accumulator = _mm256_hadd_ps(accumulator, accumulator);
    accumulator = _mm256_hadd_ps(accumulator, accumulator);

    auto sum_high = _mm256_extractf128_ps(accumulator, 1);
    auto sum_low = _mm256_castps256_ps128(accumulator);

    float total_sum = _mm_cvtss_f32(sum_low) + _mm_cvtss_f32(sum_high);

    while (data_offset < n) {
        total_sum += data[data_offset];
        ++data_offset;
    }

    return total_sum;
}
#else
auto accumulateCustomSimd(const float* data, const std::size_t n) -> float {
  if (n < 4) {
    return std::accumulate(data, data + n, 0.0F);
  }

  float32x4_t accumulator = vld1q_f32(data);
  std::size_t data_offset = 4;

  for (std::size_t i = 1; i < n / 16; ++i, data_offset += 16) {
    float32x4_t tmp = vld1q_f32(data + data_offset);

    for (std::size_t j = 1; j < 4; ++j) {
      tmp = vaddq_f32(tmp, vld1q_f32(data + data_offset + j * 4));
    }

    accumulator = vaddq_f32(accumulator, tmp);
  }

  const auto temp = vadd_f32(vget_high_f32(accumulator), vget_low_f32(accumulator));
  auto total_sum = vget_lane_f32(temp, 0) + vget_lane_f32(temp, 1);

  while (data_offset < n) {
    total_sum += data[data_offset++];
  }

  return total_sum;
}
#endif

auto accumulateOmpSimdMultithreading(std::span<const float> vec) -> float {
  float result = 0.0F;

#pragma omp parallel num_threads(ThreadsCount)
  {
#pragma omp for simd reduction(+ : result) nowait
    for (std::size_t i = 0; i < vec.size(); ++i) {
      result += vec[i];
    }
  }

  return result;
}

auto accumulateOmpMultithreading(std::span<const float> vec) -> float {
  float result = 0.0F;

  const auto length = static_cast<std::intptr_t>(vec.size());

#pragma omp parallel for num_threads(ThreadsCount) reduction(+ : result)
  for (std::intptr_t i = 0; i < length; ++i) {
    result += vec[i];
  }

  return result;
}

auto accumulateAsyncMultithreading(const std::span<const float> vec) -> isl::Task<float> {
  std::vector<isl::AsyncTask<float>> tasks;
  tasks.reserve(ThreadsCount);

  const auto chunk_size = vec.size() / ThreadsCount;
  std::ptrdiff_t begin{};
  std::ptrdiff_t end{};

  for (std::size_t i = 0; i < ThreadsCount; ++i) {
    begin = static_cast<std::ptrdiff_t>(chunk_size * (i - 1));
    end = static_cast<std::ptrdiff_t>(chunk_size * i);

    tasks.emplace_back(
      pool.async([](std::span<const float> vec)-> isl::Task<float> {
        co_return std::accumulate(vec.begin(), vec.end(), 0.0F);
      }(std::span(vec.data() + begin, vec.data() + end)))
    );
  }

  float result = 0.0F;

  for (auto&task: tasks) {
    result += co_await task;
  }

  co_return result;
}

auto accumulateSimdMultithreading(std::span<const float> vec) -> float {
  float result = 0.0F;

#pragma omp parallel default(none) shared(vec, result) num_threads(ThreadsCount)
  {
    float local_accumulator = 0.0;

    const auto thread_id = static_cast<std::size_t>(omp_get_thread_num());
    const auto total_threads = static_cast<std::size_t>(omp_get_num_threads());

    // Compute the range for each thread
    const std::size_t chunk_size = vec.size() / total_threads; // Equal chunk size
    const std::size_t start = thread_id * chunk_size;
    const std::size_t end = (thread_id == total_threads - 1) ? vec.size() : start + chunk_size;

    local_accumulator = accumulateCustomSimd(vec.data() + start, end - start);

#pragma omp critical
    result += local_accumulator;
  }

  return result;
}

template<typename T>
struct SharedFrame {
  std::atomic<std::size_t> indexGenerator{};
  std::atomic<std::size_t> completedTasksCount{};
  std::array<T, 128> result{};
};

float accumulateWithFork(std::span<const float> data, const std::size_t forks_count) {
  constexpr static std::size_t frame_size = 0x1000;

  void* shared_memory = mmap(nullptr, frame_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

  if (shared_memory == nullptr) {
    throw std::bad_alloc{};
  }

  auto* shared_frame = new(shared_memory) SharedFrame<float>{};

  const auto chunk_size = data.size() / forks_count;
  std::ptrdiff_t begin{};
  std::ptrdiff_t end{};

  for (std::size_t i = 0; i < forks_count - 1; ++i) {
    begin = static_cast<std::ptrdiff_t>(chunk_size * i);
    end = static_cast<std::ptrdiff_t>(chunk_size * (i + 1));
    const auto in_frame_index = shared_frame->indexGenerator.fetch_add(1);

    const pid_t pid = fork();

    if (pid < 0) {
      throw std::runtime_error{"fork failed"};
    }

    if (pid == 0) {
      // child
      shared_frame->result[in_frame_index] = std::accumulate(data.begin() + begin, data.begin() + end, 0.0F);
      shared_frame->completedTasksCount.fetch_add(1);
      std::exit(EXIT_SUCCESS);
    }
  }

  auto result = std::accumulate(data.begin() + end, data.end(), 0.0F);

  while (shared_frame->completedTasksCount.load() < forks_count - 1) {
    std::this_thread::yield();
  }

  for (std::size_t i = 0; i < forks_count - 1; ++i) {
    result += shared_frame->result[i];
  }

  munmap(shared_memory, frame_size);

  return result;
}

auto forkMax(std::span<const float> data, const std::size_t forks_count) -> decltype(data.begin()) {
  constexpr static std::size_t frame_size = 0x1000;

  void* shared_memory = mmap(nullptr, frame_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

  if (shared_memory == nullptr) {
    throw std::bad_alloc{};
  }

  auto* shared_frame = new(shared_memory) SharedFrame<std::ptrdiff_t>{};

  const auto chunk_size = data.size() / forks_count;
  std::ptrdiff_t start{};
  std::ptrdiff_t end{};

  for (std::size_t i = 0; i < forks_count - 1; ++i) {
    start = static_cast<std::ptrdiff_t>(chunk_size * i);
    end = static_cast<std::ptrdiff_t>(chunk_size * (i + 1));
    const auto result_index = shared_frame->indexGenerator.fetch_add(1);

    const pid_t pid = fork();

    if (pid < 0) {
      throw std::runtime_error{"fork failed"};
    }

    if (pid == 0) {
      // child
      shared_frame->result[result_index] = std::max_element(data.begin() + start, data.begin() + end) - data.
                                           begin();
      shared_frame->completedTasksCount.fetch_add(1);
      std::exit(EXIT_SUCCESS);
    }
  }

  auto result = std::max_element(data.begin() + end, data.end());

  while (shared_frame->completedTasksCount.load() < forks_count - 1) {
    std::this_thread::yield();
  }

  for (std::size_t i = 0; i < forks_count - 1; ++i) {
    if (const auto idx = shared_frame->result[i]; *(data.begin() + idx) > *result) {
      result = data.begin() + idx;
    }
  }

  munmap(shared_memory, frame_size);

  return result;
}

static void B_ompWrongParallelMax(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(ompWrongParallelMax(std::span(data.data(), state.range(0))));
  }
}

static void B_ompParallelMax(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(ompParallelMax(std::span(data.data(), state.range(0))));
  }
}

static void B_ompParallelMaxSimd(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(ompParallelMaxWithSimd(std::span(data.data(), state.range(0))));
  }
}

static void B_trivialMax(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(trivialMax(std::span(data.data(), state.range(0))));
  }
}

static void B_trivialMaxSimd(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(simdMax(data.data(), state.range(0)));
  }
}

static void B_standardMax(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(standardMax(std::span(data.data(), state.range(0))));
  }
}

static void B_forkMax(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(forkMax(data, ThreadsCount));
  }
}

#if !defined(_LIBCPP_VERSION)
static void B_standardMaxParallel(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(standardMaxParallel(data));
    }
}
#endif

static void B_accumulateOmpSimd(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(accumulateOmpSimd(std::span(data.data(), state.range(0))));
  }
}

static void B_accumulateOmpMultithreading(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(accumulateOmpMultithreading(std::span(data.data(), state.range(0))));
  }
}

static void B_accumulateAsyncMultithreading(benchmark::State&state) {
  for (auto _: state) {
    auto task = accumulateAsyncMultithreading(std::span(data.data(), state.range(0)));
    task.await();
    benchmark::DoNotOptimize(task.get());
  }
}

static void B_accumulateOmpSimdMultithreading(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(accumulateOmpSimdMultithreading(std::span(data.data(), state.range(0))));
  }
}

static void B_accumulateSimdMultithreading(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(accumulateSimdMultithreading(std::span(data.data(), state.range(0))));
  }
}

static void B_accumulateStd(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(std::accumulate(data.begin(), data.begin() + state.range(0), 0.0F));
  }
}

static void B_accumulateSimd(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(accumulateCustomSimd(data.data(), state.range(0)));
  }
}

static void B_forkAccumulate(benchmark::State&state) {
  for (auto _: state) {
    benchmark::DoNotOptimize(accumulateWithFork(std::span(data.data(), state.range(0)), ThreadsCount));
  }
}


// BENCHMARK(B_ompParallelMaxSimd);
// BENCHMARK(B_ompWrongParallelMax);
// BENCHMARK(B_trivialMax);
// BENCHMARK(B_ompParallelMax);
// BENCHMARK(B_trivialMaxSimd);
// BENCHMARK(B_standardMax);
// BENCHMARK(B_forkMax);
//
// #if !defined(_LIBCPP_VERSION)
// BENCHMARK(B_standardMaxParallel);
// #endif
//
// BENCHMARK(B_accumulateOmpSimd)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000)->Unit(
//   benchmark::kMicrosecond);
// BENCHMARK(B_accumulateStd)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000)->Unit(
//   benchmark::kMicrosecond);
// BENCHMARK(B_accumulateSimd)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000)->Unit(
// benchmark::kMicrosecond);
// BENCHMARK(B_accumulateOmpSimdMultithreading);
// BENCHMARK(B_accumulateSimdMultithreading);
// BENCHMARK(B_forkAccumulate)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000)->Iterations(300);

// BENCHMARK(B_accumulateStd)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000)->Unit(
//   benchmark::kMicrosecond);
//
BENCHMARK(B_accumulateSimd)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000)->Unit(
  benchmark::kMicrosecond);

BENCHMARK(B_accumulateSimdMultithreading)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->
Arg(10'000'000)->Unit(
      benchmark::kMicrosecond);

// BENCHMARK(B_accumulateStd)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000)->Unit(
//   benchmark::kMicrosecond);


// BENCHMARK(B_accumulateAsyncMultithreading)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->
// Arg(10'000'000)->Unit(benchmark::kMicrosecond);

// BENCHMARK(B_ompParallelMax)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->
// Arg(10'000'000)->Unit(benchmark::kMicrosecond);
//
// BENCHMARK(B_trivialMaxSimd)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->
// Arg(10'000'000)->Unit(benchmark::kMicrosecond);
//
// BENCHMARK(B_standardMax)->Arg(100)->Arg(1000)->Arg(10'000)->Arg(100'000)->Arg(1'000'000)->
// Arg(10'000'000)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();

/*
B_forkAccumulate/100/iterations:300          570564 ns       530353 ns          300
B_forkAccumulate/1000/iterations:300         588273 ns       526037 ns          300
B_forkAccumulate/10000/iterations:300        609384 ns       544510 ns          300
B_forkAccumulate/100000/iterations:300       637941 ns       592270 ns          300
B_forkAccumulate/1000000/iterations:300      857817 ns       796100 ns          300
B_forkAccumulate/10000000/iterations:300    3124520 ns      2989200 ns          300
*/

// int main() {
//   fmt::println("{}", *standardMax(data));
//   fmt::println("{}", *ompParallelMax(data));
//   fmt::println("{}", *ompParallelMaxWithSimd(data));
//   fmt::println("{}", *ompWrongParallelMax(data));
//   fmt::println("{}", *simdMax(data.data(), data.size()));
//   fmt::println("{}", *forkMax(randomNormalData, 4));
//
//   fmt::println("{}", std::accumulate(data.begin(), data.end(), 0.0F));
//   fmt::println("{}", accumulateOmpSimd(data));
//   fmt::println("{}", accumulateCustomSimd(data.data(), data.size()));
//   fmt::println("{}", accumulateOmpSimdMultithreading(data));
//   fmt::println("{}", accumulateSimdMultithreading(data));
//   // fmt::println("{}", accumulateAsyncMultithreading(data));
//   fmt::println("{}", accumulateWithFork(data, 4));
//
//   return 0;
// }
