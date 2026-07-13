#include "farthest_neighbor.h"
#include "geometry.h"
#include "../utils/kernel_utils.cuh"

namespace vorotracing
{

__global__ void farthest_neighbor_kernel(const Vec3f *__restrict__ points,
                                         const uint32_t *point_adjacency,
                                         const uint32_t *point_adjacency_offsets,
                                         uint32_t num_points,
                                         uint32_t *__restrict__ indices,
                                         float *__restrict__ cell_radius)
{
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_points)
    {
        return;
    }

    Vec3f primal_point = points[i];
    uint32_t point_adjacency_begin = point_adjacency_offsets[i];
    uint32_t point_adjacency_end = point_adjacency_offsets[i + 1];
    uint32_t num_faces = point_adjacency_end - point_adjacency_begin;
    uint32_t farthest_idx = UINT32_MAX;
    float sum_distance = 0.0f;
    float max_distance = 0.0f;

    for (uint32_t i = 0; i < num_faces; ++i)
    {
        uint32_t opposite_point_idx = point_adjacency[point_adjacency_begin + i];
        Vec3f opposite_point = points[opposite_point_idx];

        float distance = (opposite_point - primal_point).norm();
        sum_distance += 0.5 * distance;
        if (distance > max_distance)
        {
            max_distance = distance;
            farthest_idx = opposite_point_idx;
        }
    }

    indices[i] = farthest_idx;
    cell_radius[i] = num_faces > 0 ? sum_distance / num_faces : 0.0f;
}

void launch_farthest_neighbor(const Vec3f *points,
                              const uint32_t *point_adjacency,
                              const uint32_t *point_adjacency_offsets,
                              uint32_t num_points,
                              uint32_t *indices,
                              float *cell_radius,
                              const void *stream = nullptr)
{
    launch_kernel_1d<1024>(farthest_neighbor_kernel,
                           num_points,
                           stream,
                           points,
                           point_adjacency,
                           point_adjacency_offsets,
                           num_points,
                           indices,
                           cell_radius);
}

std::tuple<torch::Tensor, torch::Tensor>
farthest_neighbor(torch::Tensor points_in, torch::Tensor point_adjacency_in, torch::Tensor point_adjacency_offsets_in)
{
    uint32_t num_points = points_in.size(0);
    torch::Tensor points = points_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets = point_adjacency_offsets_in.contiguous();

    if (points.device().type() != at::kCUDA)
    {
        throw std::runtime_error("points must be on CUDA device");
    }

    std::vector<int64_t> indices_shape;

    for (int64_t i = 0; i < points.dim() - 1; i++)
    {
        indices_shape.push_back(points.size(i));
    }

    torch::Tensor indices = torch::zeros(indices_shape, torch::dtype(torch::kUInt32).device(points.device()));
    torch::Tensor cell_radius = torch::zeros(indices_shape, torch::dtype(torch::kFloat32).device(points.device()));

    if (points.scalar_type() == at::kFloat)
    {
        launch_farthest_neighbor(static_cast<const Vec3f *>(points.data_ptr()),
                                 static_cast<const uint32_t *>(point_adjacency.data_ptr()),
                                 static_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
                                 num_points,
                                 static_cast<uint32_t *>(indices.data_ptr()),
                                 static_cast<float *>(cell_radius.data_ptr()));
    }
    else
    {
        throw std::runtime_error("unsupported scalar type");
    }

    return std::make_tuple(indices, cell_radius);
}

} // namespace vorotracing
