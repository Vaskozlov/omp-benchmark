#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdio>
#include <execution>
#include <fmt/format.h>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#    include <immintrin.h>
#else
#    include <arm_neon.h>
#endif

const static std::size_t ThreadsCount = std::thread::hardware_concurrency() / 2;

std::vector<float> generateRandomVectorOfFloats(
    const std::size_t n, const float lower_bound, const float upper_bound)
{
    std::vector<float> result(n);
    std::mt19937_64 engine;
    std::uniform_real_distribution<float> distribution{lower_bound, upper_bound};

    for (auto &elem : result) {
        elem = distribution(engine);
    }

    return result;
}

std::vector<float> generateRandomVectorOfFloatsNormalDistribution(
    const std::size_t n, const float mean, const float der)
{
    std::vector<float> result(n);
    std::mt19937_64 engine;
    std::normal_distribution<float> distribution{mean, der};

    for (auto &elem : result) {
        elem = distribution(engine);
    }

    return result;
}

static constexpr std::size_t dataSize = 10'000'000;

const auto randomData = generateRandomVectorOfFloats(dataSize, -1000.0F, 1000.0F);

const auto randomNormalData =
    generateRandomVectorOfFloatsNormalDistribution(dataSize, 0.0F, 1000.0F);

const auto &data = randomNormalData;

auto ompWrongParallelMax(const std::vector<float> &vec) -> decltype(vec.begin())
{
    float maxValue = -std::numeric_limits<float>::infinity();
    std::size_t idx = 0;

#pragma omp parallel for
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (maxValue < vec[i]) {
            maxValue = vec[i];
            idx = i;
        }
    }

    return vec.begin() + idx;
}

auto ompParallelMax(const std::vector<float> &vec) -> decltype(vec.begin())
{
    float maxValue = -std::numeric_limits<float>::infinity();
    std::size_t idx = 0;

#pragma omp parallel default(none) shared(vec, maxValue, idx) num_threads(ThreadsCount)
    {
        auto localMax = maxValue;
        auto localIdx = idx;

#pragma omp for nowait
        for (std::size_t i = 0; i < vec.size(); ++i) {
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

    return vec.begin() + idx;
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

auto simdMax(const float *data, const std::size_t n) -> const float *
{
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

auto ompParallelMaxWithSimd(const std::vector<float> &vec) -> const float *
{
    const auto *it = vec.data();

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

auto trivialMax(const std::vector<float> &vec) -> decltype(vec.begin())
{
    float maxValue = -std::numeric_limits<float>::infinity();
    std::size_t idx = 0;

    for (std::size_t i = 0; i < randomData.size(); ++i) {
        if (maxValue < randomData[i]) {
            idx = i;
            maxValue = randomData[i];
        }
    }

    return vec.begin() + idx;
}

auto standartMax(const std::vector<float> &vec) -> decltype(vec.begin())
{
    return std::max_element(vec.begin(), vec.end());
}

#if !defined(_LIBCPP_VERSION)
auto standardMaxParallel(const std::vector<float> &vec) -> decltype(vec.begin())
{
    return std::max_element(std::execution::par, vec.begin(), vec.end());
}
#endif

auto accumulateOmpSimd(const std::vector<float> &vec) -> float
{
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
auto accumulateCustomSimd(const float *data, const std::size_t n) -> float
{
    if (n < 4) {
        return std::accumulate(data, data + n, 0.0F);
    }

    float32x4_t accumulator = vld1q_f32(data);
    std::size_t data_offset = 4;

    for (std::size_t i = 1; i < n / 4; ++i, data_offset += 4) {
        const float32x4_t tmp = vld1q_f32(data + data_offset);
        accumulator = vaddq_f32(accumulator, tmp);
    }

    const auto temp = vadd_f32(vget_high_f32(accumulator), vget_low_f32(accumulator));
    auto total_sum = vget_lane_f32(temp, 0) + vget_lane_f32(temp, 1);

    while (data_offset < n) {
        total_sum += data[data_offset];
        ++data_offset;
    }

    return total_sum;
}
#endif

auto accumulateOmpSimdMultithreading(const std::vector<float> &vec) -> float
{
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

auto accumulateSimdMultithreading(const std::vector<float> &vec) -> float
{
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

static void B_ompWrongParallelMax(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(ompWrongParallelMax(data));
    }
}

static void B_ompParallelMax(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(ompParallelMax(data));
    }
}

static void B_ompParallelMaxSimd(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(ompParallelMaxWithSimd(data));
    }
}

static void B_trivialMax(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(trivialMax(data));
    }
}

static void B_trivialMaxSimd(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(simdMax(data.data(), data.size()));
    }
}

static void B_standardMax(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(standartMax(data));
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

static void B_accumulateOmpSimd(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(accumulateOmpSimd(data));
    }
}

static void B_accumulateOmpSimdMultithreading(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(accumulateOmpSimdMultithreading(data));
    }
}

static void B_accumulateSimdMultithreading(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(accumulateSimdMultithreading(data));
    }
}

static void B_accumulateStd(benchmark::State &state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(std::accumulate(data.begin(), data.end(), 0.0F));
    }
}

BENCHMARK(B_ompParallelMax);
BENCHMARK(B_ompParallelMaxSimd);
BENCHMARK(B_ompWrongParallelMax);
BENCHMARK(B_trivialMax);
BENCHMARK(B_trivialMaxSimd);
BENCHMARK(B_standardMax);

#if !defined(_LIBCPP_VERSION)
BENCHMARK(B_standardMaxParallel);
#endif

BENCHMARK(B_accumulateOmpSimd);
BENCHMARK(B_accumulateStd);
BENCHMARK(B_accumulateOmpSimdMultithreading);
BENCHMARK(B_accumulateSimdMultithreading);

BENCHMARK_MAIN();

// int main() {
//   fmt::println("{}", *standartMax(randomNormalData));
//   fmt::println("{}", *ompParallelMax(randomNormalData));
//   fmt::println("{}", *ompParallelMaxWithSimd(randomNormalData));
//   fmt::println("{}", *simdMax(randomNormalData.data(), randomNormalData.size()));

//   fmt::println("{}", std::accumulate(randomNormalData.begin(), randomNormalData.end(), 0.0F));
//   fmt::println("{}", accumulateOmpSimd(randomNormalData));
//   fmt::println("{}", accumulateCustomSimd(randomNormalData.data(), randomNormalData.size()));
//   fmt::println("{}", accumulateOmpSimdMultithreading(randomNormalData));
//   fmt::println("{}", accumulateSimdMultithreading(randomNormalData));
// }
