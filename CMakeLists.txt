cmake_minimum_required(VERSION 3.20)
project(omp-test LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(GPU_ARCH "sm_60")
set(CMAKE_CUDA_ARCHITECTURES 60)

include(cmake/CPM.cmake)

enable_language(CUDA)

# set(CMAKE_LINKER clang++-20)
# add_link_options("-fuse-ld=lld")

find_package(OpenMP REQUIRED)
find_package(benchmark REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit)


CPMAddPackage(
        NAME fmt
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 577fd3be883accf8629423ed77fcca8a24bccee2
)


add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX fmt::fmt benchmark::benchmark TBB::tbb Threads::Threads CUDA::cudart CUDA::cudart)

target_link_options(${PROJECT_NAME} PUBLIC -fopenmp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_link_options(${PROJECT_NAME} PUBLIC -fopenmp -fopenmp-targets=nvptx64 --offload-arch=sm_70 --libomptarget-nvptx-bc-path=/usr/lib/llvm-20/lib/libomptarget-nvptx.bc)
        target_compile_options(${PROJECT_NAME} PUBLIC -mavx2 -fopenmp -fopenmp-targets=nvptx64 --offload-arch=sm_70 --libomptarget-nvptx-bc-path=/usr/lib/llvm-20/lib/libomptarget-nvptx.bc)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${PROJECT_NAME} PUBLIC -mavx2 -foffload=nvptx-none -march=native -fcf-protection=none -fno-stack-protector)
endif()


