cmake_minimum_required(VERSION 3.20)
project(omp-test)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CPM.cmake)

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
        GIT_TAG f9d13f7d63ac7160962746a17788a60d937f0517
)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX fmt::fmt benchmark::benchmark TBB::tbb isl Threads::Threads)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    target_compile_options(${PROJECT_NAME} PRIVATE "-mavx2")
endif()
