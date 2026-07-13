#pragma once

#include <functional>
#include <sstream>
#include <stdexcept>

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

// For backtrace support
#include <cxxabi.h>
#include <execinfo.h>

namespace vorotracing
{

inline std::string get_backtrace(int skip = 1)
{
    const int max_frames = 64;
    void *buffer[max_frames];
    int num_frames = backtrace(buffer, max_frames);
    char **symbols = backtrace_symbols(buffer, num_frames);

    std::ostringstream oss;
    oss << "\nCall stack:\n";

    for (int i = skip; i < num_frames; ++i)
    {
        // Try to demangle C++ symbol names
        char *mangled_name = nullptr;
        char *offset_begin = nullptr;
        char *offset_end = nullptr;

        // Find the mangled name
        for (char *p = symbols[i]; *p; ++p)
        {
            if (*p == '(')
                mangled_name = p;
            else if (*p == '+')
                offset_begin = p;
            else if (*p == ')')
            {
                offset_end = p;
                break;
            }
        }

        if (mangled_name && offset_begin && offset_end && mangled_name < offset_begin)
        {
            *mangled_name++ = '\0';
            *offset_begin++ = '\0';
            *offset_end = '\0';

            int status;
            char *demangled = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

            if (status == 0)
            {
                oss << "  [" << i - skip << "] " << symbols[i] << " : " << demangled << " + " << offset_begin << "\n";
                free(demangled);
            }
            else
            {
                oss << "  [" << i - skip << "] " << symbols[i] << " : " << mangled_name << " + " << offset_begin
                    << "\n";
            }
        }
        else
        {
            oss << "  [" << i - skip << "] " << symbols[i] << "\n";
        }
    }

    free(symbols);
    return oss.str();
}

inline void cuda_check_fn(cudaError_t err, int line, const char *file)
{
    if (err != cudaSuccess)
    {
        std::string msg = "CUDA call at " + std::string(file) + ":" + std::to_string(line) + " failed: ";
        msg = msg + cudaGetErrorString(err);
        msg += get_backtrace(2); // Skip this function and the macro
        throw std::runtime_error(msg);
    }
}

inline void cuda_check_fn(CUresult err, int line, const char *file)
{
    if (err != CUDA_SUCCESS)
    {
        const char *msg;
        cuGetErrorString(err, &msg);
        std::string error_msg = std::string("CUDA call at ") + file + ":" + std::to_string(line) + " failed: " + msg;
        error_msg += get_backtrace(2); // Skip this function and the macro
        throw std::runtime_error(error_msg);
    }
}

#define cuda_check(call) vorotracing::cuda_check_fn(call, __LINE__, __FILE__)

inline void global_cuda_init()
{
    cuda_check(cuInit(0));
    cuda_check(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1ul << 29ul));
}

inline void set_default_stream()
{
    auto stream = at::cuda::getCurrentCUDAStream();
    at::cuda::setCurrentCUDAStream(stream);
}

} // namespace vorotracing