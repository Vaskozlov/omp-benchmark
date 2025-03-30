cmake_minimum_required(VERSION 3.20)
project(omp-test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)
find_package(benchmark REQUIRED)
find_package(fmt REQUIRED)
find_package(TBB REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX fmt::fmt benchmark::benchmark TBB::tbb)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    target_compile_options(${PROJECT_NAME} PRIVATE "-mavx2")
endif()
