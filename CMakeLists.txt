cmake_minimum_required(VERSION 3.20)
project(omp-test)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(GPU_ARCH "sm_60")
set(CMAKE_CUDA_ARCHITECTURES 60)

include(cmake/CPM.cmake)

enable_language(CUDA)

find_package(OpenMP REQUIRED)
find_package(benchmark REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)


CPMAddPackage(
        NAME fmt
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 577fd3be883accf8629423ed77fcca8a24bccee2
)

CPMAddPackage(
        NAME isl
        GITHUB_REPOSITORY Vaskozlov/ISL
        GIT_TAG 38eb61255b467d095a9d3c7137f4dd5f6f6e8308
)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX fmt::fmt benchmark::benchmark TBB::tbb Threads::Threads)


target_compile_options(${PROJECT_NAME} PRIVATE "-foffload=nvptx-none")
target_compile_options(${PROJECT_NAME} PRIVATE "-mavx2")

