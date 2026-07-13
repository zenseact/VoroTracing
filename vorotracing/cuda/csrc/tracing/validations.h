#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <torch/torch.h>

#include "../utils/cuda_helpers.h"

namespace vorotracing
{

inline void validate_scene_data(torch::Tensor points,
                         torch::Tensor attributes,
                         torch::Tensor point_adjacency,
                         torch::Tensor point_adjacency_offsets)
{

    if (points.size(-1) != 3)
    {
        throw std::runtime_error("points had dimension " + std::to_string(points.size(-1)) +
                                 " along axis -1, expected 3");
    }
    if (points.scalar_type() != at::kFloat)
    {
        throw std::runtime_error("points had dtype " + std::string(c10::toString(points.scalar_type())) +
                                 ", expected " + std::string("float32"));
    }
    if (points.device().type() != at::kCUDA)
    {
        throw std::runtime_error("points must be on CUDA device");
    }
    uint32_t num_points = points.numel() / 3;

    if (attributes.device().type() != at::kCUDA)
    {
        throw std::runtime_error("attributes must be on CUDA device");
    }

    if (point_adjacency_offsets.scalar_type() != at::kUInt32)
    {
        throw std::runtime_error("point_adjacency_offsets must have uint32 dtype");
    }
    if (point_adjacency_offsets.device().type() != at::kCUDA)
    {
        throw std::runtime_error("point_adjacency_offsets must be on CUDA device");
    }
    if (point_adjacency_offsets.numel() != num_points + 1)
    {
        throw std::runtime_error("point_adjacency_offsets must have num_points "
                                 "+ 1 elements");
    }

    if (point_adjacency.scalar_type() != at::kUInt32)
    {
        throw std::runtime_error("point_adjacency must have uint32 dtype");
    }
    if (point_adjacency.device().type() != at::kCUDA)
    {
        throw std::runtime_error("point_adjacency must be on CUDA device");
    }
}

} // namespace vorotracing