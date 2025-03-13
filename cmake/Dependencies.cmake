include(FetchContent)

if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(GTest QUIET)
endif()
if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    option(BUILD_GTEST "Builds the googletest subproject" ON)
    option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
    option(INSTALL_GTEST "Enable installation of googletest." OFF)
    if(EXISTS /usr/src/googletest AND NOT DEPENDENCIES_FORCE_DOWNLOAD)
        FetchContent_Declare(
        googletest
        SOURCE_DIR /usr/src/googletest
        )
    else()
        message(STATUS "Google Test not found. Fetching...")
        FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
        )
    endif()
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main  ALIAS gtest_main)
else()
    find_package(GTest REQUIRED)
    if(TARGET GTest::gtest_main AND NOT TARGET GTest::Main)
        add_library(GTest::GTest ALIAS GTest::gtest)
        add_library(GTest::Main  ALIAS GTest::gtest_main)
    endif()
endif()

if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(benchmark CONFIG QUIET)
endif()
if(NOT TARGET benchmark::benchmark)
    message(STATUS "Google Benchmark not found. Fetching...")
    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
    option(BENCHMARK_ENABLE_INSTALL "Enable installation of benchmark." OFF)
    FetchContent_Declare(
        googlebench
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
    )
    set(HAVE_STD_REGEX ON)
    set(RUN_HAVE_STD_REGEX 1)
    FetchContent_MakeAvailable(googlebench)
    if(NOT TARGET benchmark::benchmark)
        add_library(benchmark::benchmark ALIAS benchmark)
    endif()
else()
    find_package(benchmark CONFIG REQUIRED)
endif()
