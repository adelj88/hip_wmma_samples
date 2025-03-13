# Create an executable target
add_executable(test test.cpp)

# Include HIP include directories
target_include_directories(test PRIVATE ${HIP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link HIP libraries
target_link_libraries(test PRIVATE ${HIP_LIBRARIES} roc::rocblas GTest::GTest GTest::Main)

enable_testing()
add_test(NAME hgemm_test COMMAND test)

# Create an executable target
add_executable(bench bench.cpp)

# Include HIP include directories
target_include_directories(bench PRIVATE ${HIP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link HIP libraries
target_link_libraries(bench PRIVATE ${HIP_LIBRARIES} roc::rocblas benchmark::benchmark)
