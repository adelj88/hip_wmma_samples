/*
 * MIT License
 *
 * Copyright (c) 2024 Adel Johar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef HIP_UTILS_HPP
#define HIP_UTILS_HPP

#include <hip/hip_runtime.h>
#include <iostream>

/**
 * @brief Macro for checking HIP errors
 * @param call HIP API call to check
 */
#define HIP_CHECK(call)                                                                        \
    {                                                                                          \
        hipError_t status = call;                                                              \
        if(status != hipSuccess)                                                               \
        {                                                                                      \
            std::cerr << "HIP error: " #call " failed with error " << static_cast<int>(status) \
                      << ": " << hipGetErrorString(status) << std::endl;                       \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    }

/**
 * @brief Timer class for performance measurements
 */
class gpu_timer
{
public:
    gpu_timer()
    {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }

    ~gpu_timer()
    {
        HIP_CHECK(hipEventDestroy(start_));
        HIP_CHECK(hipEventDestroy(stop_));
    }

    void start(hipStream_t& stream)
    {
        HIP_CHECK(hipEventRecord(start_, stream));
    }

    float stop(hipStream_t& stream)
    {
        HIP_CHECK(hipEventRecord(stop_, stream));
        HIP_CHECK(hipEventSynchronize(stop_));
        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start_, stop_));
        return elapsed;
    }

private:
    hipEvent_t start_, stop_;
};

#endif // HIP_UTILS_HPP
