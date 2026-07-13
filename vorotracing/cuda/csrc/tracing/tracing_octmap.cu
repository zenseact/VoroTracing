#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdint.h>
#include <tuple>

#include "../utils/cuda_array.h"
#include "../utils/cuda_helpers.h"
#include "../utils/geometry.h"
#include "../utils/kernel_utils.cuh"
#include "../utils/typing.cuh"
#include "camera.h"
#include "ray_integrator.cuh"
#include "ray_integrator_octmap.cuh"
#include "validations.h"

#include "tracing.h"
#include "tracing_octmap.h"

#define ASSERT(condition, message)                                                                                     \
    if (!(condition))                                                                                                  \
    {                                                                                                                  \
        throw std::runtime_error(message);                                                                             \
    }

constexpr uint32_t OCTMAP_RES = 8;
constexpr uint32_t OCTMAP_RES_SPEC = 8;
constexpr uint32_t BLOCK_SIZE_OCTMAP = 128;

namespace vorotracing
{

// trace_octmap is octmap-specific, so it takes
// the integrator by concrete type rather than via the RayIntegrator<> base
// class. The integrator stores its own per-cell attribute pointers, so the
// only "data" the loop hands it is the optional per-point debug-output buffer
// (point_contribution in fwd, point_error in bwd).
template <int chunk_size, typename Integrator, typename debug_scalar>
__device__ uint32_t trace_octmap(const Ray &ray,
                                 const Vec3f *__restrict__ points,
                                 const uint32_t *__restrict__ point_adjacency,
                                 const uint32_t *__restrict__ point_adjacency_offsets,
                                 const Vec4h *__restrict__ adjacent_points,
                                 debug_scalar *__restrict__ debug_output,
                                 uint32_t start_point,
                                 uint32_t max_steps,
                                 Integrator &ray_integrator)
{
    float t_0 = 0.0f;
    uint32_t n = 0;

    uint32_t current_point_idx = start_point;
    Vec3f primal_point = points[current_point_idx];

    for (;;)
    {
        n++;
        if (n > max_steps)
        {
            break;
        }

        uint32_t point_adjacency_begin = point_adjacency_offsets[current_point_idx];
        uint32_t point_adjacency_end = point_adjacency_offsets[current_point_idx + 1];

        uint32_t num_faces = point_adjacency_end - point_adjacency_begin;
        float t_1 = std::numeric_limits<float>::infinity();

        uint32_t next_face = UINT32_MAX;
        Vec3f next_point = Vec3f::Zero();

        half2 chunk[chunk_size * 2];
        for (uint32_t i = 0; i < num_faces; i += chunk_size)
        {
#pragma unroll
            for (uint32_t j = 0; j < chunk_size; ++j)
            {
                chunk[2 * j] = reinterpret_cast<const half2 *>(adjacent_points + point_adjacency_begin + i + j)[0];
                chunk[2 * j + 1] = reinterpret_cast<const half2 *>(adjacent_points + point_adjacency_begin + i + j)[1];
            }

#pragma unroll
            for (uint32_t j = 0; j < chunk_size; ++j)
            {
                Vec3f offset(__half2float(chunk[2 * j].x),
                             __half2float(chunk[2 * j].y),
                             __half2float(chunk[2 * j + 1].x));
                Vec3f face_origin = primal_point + offset / 2.0f;
                Vec3f face_normal = offset;
                float dp = face_normal.dot(ray.direction);
                float t = (face_origin - ray.origin).dot(face_normal) / dp;

                if (dp > 0.0f && t < t_1 && (i + j) < num_faces)
                {
                    t_1 = t;
                    next_face = i + j;
                }
            }
        }

        if (next_face == UINT32_MAX)
        {
            break;
        }

        uint32_t next_point_idx = point_adjacency[point_adjacency_begin + next_face];
        next_point = points[next_point_idx];

        if (t_1 > t_0)
        {
            if (!ray_integrator.integrate_cell(current_point_idx, t_0, t_1, primal_point, next_point, debug_output))
            {
                break;
            }
        }
        t_0 = fmaxf(t_0, t_1);
        current_point_idx = next_point_idx;
        primal_point = next_point;
    }

    return n;
}

__global__ void prefetch_adjacent_diff_octmap_kernel(const Vec3f *__restrict__ points,
                                                     uint32_t num_points,
                                                     uint32_t point_adjacency_size,
                                                     const uint32_t *__restrict__ point_adjacency,
                                                     const uint32_t *__restrict__ point_adjacency_offsets,
                                                     Vec4h *__restrict__ adjacent_diff)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    Vec3f p = points[i];
    uint32_t offset_start = point_adjacency_offsets[i];
    uint32_t offset_end = point_adjacency_offsets[i + 1];
    uint32_t num_adjacent = offset_end - offset_start;

    for (uint32_t j = 0; j < num_adjacent; ++j)
    {
        uint32_t adjacent_idx = point_adjacency[offset_start + j];
        Vec3f q = points[adjacent_idx];
        Vec3f diff = q - p;
        adjacent_diff[offset_start + j] =
            Vec4h(__float2half(diff[0]), __float2half(diff[1]), __float2half(diff[2]), __float2half(0.0f));
    }
}

template <typename attr_scalar, int diff_map_res, int spec_map_res>
__global__ void trace_vorotracing_fwd_kernel(TraceSettings settings,
                                             const Vec3f *__restrict__ points,
                                             const attr_scalar *__restrict__ diffuse,
                                             const attr_scalar *__restrict__ specular,
                                             const float *__restrict__ density,
                                             const uint32_t *__restrict__ point_adjacency,
                                             const uint32_t *__restrict__ point_adjacency_offsets,
                                             const Vec4h *__restrict__ adjacent_diff,
                                             const Ray *__restrict__ rays,
                                             uint32_t num_rays,
                                             const uint32_t *__restrict__ start_point_index,
                                             uint32_t num_depth_quantiles,
                                             const float *__restrict__ depth_quantiles,
                                             attr_scalar *__restrict__ ray_rgba,
                                             float *__restrict__ quantile_depths,
                                             uint32_t *__restrict__ quantile_point_indices,
                                             uint32_t *__restrict__ num_intersections,
                                             attr_scalar *__restrict__ point_contribution,
                                             float *__restrict__ distortion_out,
                                             float *__restrict__ W_total_out,
                                             float *__restrict__ S_total_out,
                                             float *__restrict__ t_far_out)
{
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    const float *ray_depth_quantiles = depth_quantiles + thread_idx * num_depth_quantiles;

    uint32_t start_point = start_point_index[thread_idx];

    RayIntegratorOctMap<attr_scalar, diff_map_res, spec_map_res> ray_integrator(
        ray,
        diffuse,
        specular,
        density,
        settings.weight_threshold,
        ray_depth_quantiles,
        depth_quantiles ? num_depth_quantiles : 0,
        quantile_depths + thread_idx * num_depth_quantiles,
        quantile_point_indices + thread_idx * num_depth_quantiles);

    uint32_t n = trace_octmap<4>(ray,
                                 points,
                                 point_adjacency,
                                 point_adjacency_offsets,
                                 adjacent_diff,
                                 point_contribution,
                                 start_point,
                                 settings.max_intersections,
                                 ray_integrator);

    Vec3f accumulated_rgb = ray_integrator.get_accumulated_rgb();
    float transmittance = ray_integrator.get_transmittance();
    uint32_t num_filled_quantiles = ray_integrator.get_num_filled_quantiles();

    for (uint32_t i = num_filled_quantiles; i < num_depth_quantiles; ++i)
    {
        quantile_depths[thread_idx * num_depth_quantiles + i] = -1.0f;
        quantile_point_indices[thread_idx * num_depth_quantiles + i] = UINT32_MAX;
    }

    for (uint32_t i = 0; i < 3; ++i)
    {
        ray_rgba[thread_idx * 4 + i] = from_float<attr_scalar>(accumulated_rgb[i]);
    }
    ray_rgba[thread_idx * 4 + 3] = from_float<attr_scalar>(1 - transmittance);

    if (num_intersections)
        num_intersections[thread_idx] = n;

    distortion_out[thread_idx] = ray_integrator.get_distortion();
    W_total_out[thread_idx] = ray_integrator.get_W_total();
    S_total_out[thread_idx] = ray_integrator.get_S_total();
    t_far_out[thread_idx] = ray_integrator.get_t_far();
}

template <typename attr_scalar, int diff_map_res, int spec_map_res>
__global__ void trace_vorotracing_bwd_kernel(TraceSettings settings,
                                             const Vec3f *__restrict__ points,
                                             const attr_scalar *__restrict__ diffuse,
                                             const attr_scalar *__restrict__ specular,
                                             const float *__restrict__ density,
                                             const uint32_t *__restrict__ point_adjacency,
                                             const uint32_t *__restrict__ point_adjacency_offsets,
                                             const Vec4h *__restrict__ adjacent_diff,
                                             const Ray *__restrict__ rays,
                                             uint32_t num_rays,
                                             const uint32_t *__restrict__ start_point_index,
                                             uint32_t num_depth_quantiles,
                                             const float *__restrict__ depth_quantiles,
                                             const uint32_t *__restrict__ quantile_point_indices,
                                             const attr_scalar *__restrict__ ray_rgba,
                                             const attr_scalar *__restrict__ ray_rgba_grad,
                                             const float *__restrict__ depth_grad,
                                             const attr_scalar *__restrict__ ray_error,
                                             float contribution_alpha_grad,
                                             const float *__restrict__ distortion_in,
                                             const float *__restrict__ distortion_grad,
                                             const float *__restrict__ W_total_in,
                                             const float *__restrict__ S_total_in,
                                             const float *__restrict__ t_far_in,
                                             Ray *__restrict__ ray_grad,
                                             Vec3f *__restrict__ points_grad,
                                             attr_scalar *__restrict__ diffuse_grad,
                                             attr_scalar *__restrict__ specular_grad,
                                             float *__restrict__ density_grad,
                                             attr_scalar *__restrict__ point_error)
{
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    const float *ray_depth_grad = depth_grad + thread_idx * num_depth_quantiles;
    const float *ray_depth_quantiles = depth_quantiles + thread_idx * num_depth_quantiles;

    Vec4f rgba_grad, rgba;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
    {
        rgba_grad[i] = to_float(ray_rgba_grad[thread_idx * 4 + i]);
        rgba[i] = to_float(ray_rgba[thread_idx * 4 + i]);
    }
    rgba_grad[3] += contribution_alpha_grad;

    float error;
    if (ray_error)
    {
        error = to_float(ray_error[thread_idx]);
    }

    float current_depth_grad = 0.0f;
    for (uint32_t i = 0; i < num_depth_quantiles; ++i)
    {
        if (quantile_point_indices[thread_idx * num_depth_quantiles + i] != UINT32_MAX)
        {
            uint32_t point_idx = quantile_point_indices[thread_idx * num_depth_quantiles + i];
            float s = density[point_idx];
            current_depth_grad += ray_depth_grad[i] / s;
        }
    }

    uint32_t start_point = start_point_index[thread_idx];

    float fwd_W_total = W_total_in ? W_total_in[thread_idx] : 0.0f;
    float fwd_S_total = S_total_in ? S_total_in[thread_idx] : 0.0f;
    float fwd_t_far = t_far_in ? t_far_in[thread_idx] : 0.0f;
    float fwd_distortion = distortion_in ? distortion_in[thread_idx] : 0.0f;
    float fwd_distortion_grad = distortion_grad ? distortion_grad[thread_idx] : 0.0f;

    RayIntegratorOctMapBackward<attr_scalar, diff_map_res, spec_map_res> ray_integrator(
        ray,
        diffuse,
        specular,
        density,
        settings.weight_threshold,
        rgba,
        rgba_grad,
        ray_depth_grad,
        ray_depth_quantiles,
        depth_quantiles ? num_depth_quantiles : 0,
        current_depth_grad,
        error,
        fwd_W_total,
        fwd_S_total,
        fwd_t_far,
        fwd_distortion,
        fwd_distortion_grad,
        points_grad,
        diffuse_grad,
        specular_grad,
        density_grad);

    trace_octmap<4>(ray,
                    points,
                    point_adjacency,
                    point_adjacency_offsets,
                    adjacent_diff,
                    point_error,
                    start_point,
                    settings.max_intersections,
                    ray_integrator);
}

////////////////////////////////////////////////////////////
// Forward entry point
////////////////////////////////////////////////////////////

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
trace_vorotracing_fwd(torch::Tensor points_in,
                      torch::Tensor diffuse_in,
                      torch::Tensor specular_in,
                      torch::Tensor density_in,
                      torch::Tensor point_adjacency_in,
                      torch::Tensor point_adjacency_offsets_in,
                      torch::Tensor rays_in,
                      torch::Tensor start_point_in,
                      std::optional<torch::Tensor> depth_quantiles_in,
                      float weight_threshold,
                      uint32_t max_intersections,
                      bool return_contribution)
{
    torch::Tensor points = points_in.contiguous();
    torch::Tensor diffuse = diffuse_in.contiguous();
    torch::Tensor specular = specular_in.contiguous();
    torch::Tensor density = density_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets = point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();

    validate_scene_data(points_in, diffuse_in, point_adjacency_in, point_adjacency_offsets_in);

    bool return_depth = depth_quantiles_in.has_value();

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);
    uint32_t num_rays = rays.numel() / 6;
    uint32_t num_depth_quantiles = 0;

    ASSERT(diffuse.scalar_type() == specular.scalar_type(), "diffuse and specular must have the same dtype");
    ASSERT(diffuse.size(0) == specular.size(0) && diffuse.size(0) == density.size(0) &&
               diffuse.size(0) == (int64_t)num_points,
           "diffuse, specular, density must each have num_points rows");
    ASSERT(density.scalar_type() == at::kFloat, "density must have float32 dtype");

    ASSERT(rays.size(-1) == 6, "rays must have 6 as the last dimension");
    ASSERT(rays.scalar_type() == at::kFloat, "rays must have float32 dtype");
    ASSERT(rays.device().type() == at::kCUDA, "rays must be on CUDA device");

    ASSERT(start_point.numel() == num_rays, "start_point must have the same batch size as rays");
    ASSERT(start_point.scalar_type() == at::kUInt32, "start_point must have uint32 dtype");
    ASSERT(start_point.device().type() == at::kCUDA, "start_point must be on CUDA device");

    torch::Tensor depth_quantiles;
    if (return_depth)
    {
        depth_quantiles = depth_quantiles_in.value().contiguous();
        num_depth_quantiles = depth_quantiles.size(-1);

        ASSERT(depth_quantiles.scalar_type() == at::kFloat, "depth_quantiles must have float32 dtype");
        ASSERT(depth_quantiles.device().type() == at::kCUDA, "depth_quantiles must be on CUDA device");
        ASSERT(depth_quantiles.numel() / num_depth_quantiles == num_rays,
               "depth_quantiles must have the same batch size as rays");
    }

    TraceSettings settings = default_trace_settings();
    settings.weight_threshold = weight_threshold;
    settings.max_intersections = max_intersections;

    std::vector<int64_t> output_shape;
    for (int i = 0; i < rays.dim() - 1; i++)
    {
        output_shape.push_back(rays.size(i));
    }
    auto output_rgba_shape = output_shape;
    output_rgba_shape.push_back(4);
    torch::Tensor output_rgba =
        torch::empty(output_rgba_shape, torch::TensorOptions().dtype(diffuse.dtype()).device(rays.device()));

    auto output_num_intersections_shape = output_shape;
    output_num_intersections_shape.push_back(1);
    torch::Tensor num_intersections_out =
        torch::empty(output_num_intersections_shape,
                     torch::TensorOptions().dtype(torch::kUInt32).device(rays.device()));

    torch::Tensor output_contribution;
    if (return_contribution)
    {
        output_contribution =
            torch::zeros({num_points, 1}, torch::TensorOptions().dtype(diffuse.dtype()).device(rays.device()));
    }

    auto output_depth_shape = output_shape;
    output_depth_shape.push_back(num_depth_quantiles);
    torch::Tensor output_depth;
    torch::Tensor output_depth_indices;
    if (return_depth)
    {
        output_depth =
            torch::zeros(output_depth_shape, torch::TensorOptions().dtype(torch::kFloat32).device(rays.device()));
        output_depth_indices =
            torch::zeros(output_depth_shape, torch::TensorOptions().dtype(torch::kUInt32).device(rays.device()));
    }

    // Distortion loss outputs (per-ray scalars). Always allocated; cost is ~4 fma/cell.
    auto output_distortion_shape = output_shape;
    output_distortion_shape.push_back(1);
    auto distortion_options = torch::TensorOptions().dtype(torch::kFloat32).device(rays.device());
    torch::Tensor output_distortion = torch::empty(output_distortion_shape, distortion_options);
    torch::Tensor output_W_total = torch::empty(output_distortion_shape, distortion_options);
    torch::Tensor output_S_total = torch::empty(output_distortion_shape, distortion_options);
    torch::Tensor output_t_far = torch::empty(output_distortion_shape, distortion_options);

    auto stream = at::cuda::getCurrentCUDAStream();
    at::cuda::setCurrentCUDAStream(stream);

    CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
    launch_kernel_1d<256>(prefetch_adjacent_diff_octmap_kernel,
                          num_points,
                          nullptr,
                          reinterpret_cast<const Vec3f *>(points.data_ptr()),
                          num_points,
                          point_adjacency_size,
                          reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
                          reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
                          adjacent_diff.begin());

    if (diffuse.scalar_type() == torch::kFloat32)
    {
        launch_kernel_1d<BLOCK_SIZE_OCTMAP>(
            trace_vorotracing_fwd_kernel<float, OCTMAP_RES, OCTMAP_RES_SPEC>,
            num_rays,
            nullptr,
            settings,
            reinterpret_cast<const Vec3f *>(points.data_ptr()),
            reinterpret_cast<const float *>(diffuse.data_ptr()),
            reinterpret_cast<const float *>(specular.data_ptr()),
            reinterpret_cast<const float *>(density.data_ptr()),
            reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
            reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
            adjacent_diff.begin(),
            reinterpret_cast<const Ray *>(rays.data_ptr()),
            num_rays,
            reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
            num_depth_quantiles,
            return_depth ? reinterpret_cast<const float *>(depth_quantiles.data_ptr()) : nullptr,
            reinterpret_cast<float *>(output_rgba.data_ptr()),
            return_depth ? reinterpret_cast<float *>(output_depth.data_ptr()) : nullptr,
            return_depth ? reinterpret_cast<uint32_t *>(output_depth_indices.data_ptr()) : nullptr,
            reinterpret_cast<uint32_t *>(num_intersections_out.data_ptr()),
            return_contribution ? reinterpret_cast<float *>(output_contribution.data_ptr()) : nullptr,
            reinterpret_cast<float *>(output_distortion.data_ptr()),
            reinterpret_cast<float *>(output_W_total.data_ptr()),
            reinterpret_cast<float *>(output_S_total.data_ptr()),
            reinterpret_cast<float *>(output_t_far.data_ptr()));
    }
    else if (diffuse.scalar_type() == torch::kFloat16)
    {
        launch_kernel_1d<BLOCK_SIZE_OCTMAP>(
            trace_vorotracing_fwd_kernel<__half, OCTMAP_RES, OCTMAP_RES_SPEC>,
            num_rays,
            nullptr,
            settings,
            reinterpret_cast<const Vec3f *>(points.data_ptr()),
            reinterpret_cast<const __half *>(diffuse.data_ptr()),
            reinterpret_cast<const __half *>(specular.data_ptr()),
            reinterpret_cast<const float *>(density.data_ptr()),
            reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
            reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
            adjacent_diff.begin(),
            reinterpret_cast<const Ray *>(rays.data_ptr()),
            num_rays,
            reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
            num_depth_quantiles,
            return_depth ? reinterpret_cast<const float *>(depth_quantiles.data_ptr()) : nullptr,
            reinterpret_cast<__half *>(output_rgba.data_ptr()),
            return_depth ? reinterpret_cast<float *>(output_depth.data_ptr()) : nullptr,
            return_depth ? reinterpret_cast<uint32_t *>(output_depth_indices.data_ptr()) : nullptr,
            reinterpret_cast<uint32_t *>(num_intersections_out.data_ptr()),
            return_contribution ? reinterpret_cast<__half *>(output_contribution.data_ptr()) : nullptr,
            reinterpret_cast<float *>(output_distortion.data_ptr()),
            reinterpret_cast<float *>(output_W_total.data_ptr()),
            reinterpret_cast<float *>(output_S_total.data_ptr()),
            reinterpret_cast<float *>(output_t_far.data_ptr()));
    }
    else
    {
        throw std::runtime_error("Unsupported color dtype. Only float32 and float16 are supported.");
    }

    return std::make_tuple(output_rgba,
                           output_depth,
                           output_depth_indices,
                           output_contribution,
                           num_intersections_out,
                           output_distortion,
                           output_W_total,
                           output_S_total,
                           output_t_far);
}

////////////////////////////////////////////////////////////
// Inference entry point (slim, fp16-only)
////////////////////////////////////////////////////////////

template <int diff_map_res, int spec_map_res>
__global__ void trace_vorotracing_infer_kernel(TraceSettings settings,
                                               const Vec3f *__restrict__ points,
                                               const __half *__restrict__ diffuse,
                                               const __half *__restrict__ specular,
                                               const float *__restrict__ density,
                                               const uint32_t *__restrict__ point_adjacency,
                                               const uint32_t *__restrict__ point_adjacency_offsets,
                                               const Vec4h *__restrict__ adjacent_diff,
                                               const Ray *__restrict__ rays,
                                               uint32_t num_rays,
                                               const uint32_t *__restrict__ start_point_index,
                                               __half *__restrict__ ray_rgba,
                                               const uint32_t *__restrict__ ray_perm)
{
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    // Warp-coherence tiling: thread_idx (warp membership) follows the tile order;
    // ray_id is the actual pixel this thread handles. I/O goes to ray_id so the
    // output stays in image order (no separate permute/un-permute pass).
    uint32_t ray_id = (ray_perm != nullptr) ? ray_perm[thread_idx] : thread_idx;

    Ray ray = rays[ray_id];
    ray.direction /= ray.direction.norm();

    uint32_t start_point = start_point_index[ray_id];

    RayIntegratorOctMapInfer<__half, diff_map_res, spec_map_res> ray_integrator(ray,
                                                                                diffuse,
                                                                                specular,
                                                                                density,
                                                                                settings.weight_threshold,
                                                                                settings.cell_skip_threshold);

    // chunk_size=1: faster than 4 across all 11 scenes (+3.1% mean, 11/11).
    // Consecutive face vectors are contiguous (already coalesced), so batching
    // adds no MLP but inflates registers and issues wasted masked loads past
    // num_faces; chunk=1 does exactly num_faces loads at minimal reg pressure.
    trace_octmap<1>(ray,
                    points,
                    point_adjacency,
                    point_adjacency_offsets,
                    adjacent_diff,
                    (float *)nullptr,
                    start_point,
                    settings.max_intersections,
                    ray_integrator);

    Vec3f accumulated_rgb = ray_integrator.get_accumulated_rgb();
    float transmittance = ray_integrator.get_transmittance();

    for (uint32_t i = 0; i < 3; ++i)
    {
        ray_rgba[ray_id * 4 + i] = __float2half(accumulated_rgb[i]);
    }
    ray_rgba[ray_id * 4 + 3] = __float2half(1 - transmittance);
}

torch::Tensor trace_vorotracing_prefetch_adj(torch::Tensor points_in,
                                             torch::Tensor point_adjacency_in,
                                             torch::Tensor point_adjacency_offsets_in)
{
    torch::Tensor points = points_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets = point_adjacency_offsets_in.contiguous();

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);

    torch::Tensor adj_diff = torch::empty({(int64_t)(point_adjacency_size + 32), 4},
                                          torch::TensorOptions().dtype(torch::kFloat16).device(points.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    at::cuda::setCurrentCUDAStream(stream);

    launch_kernel_1d<256>(prefetch_adjacent_diff_octmap_kernel,
                          num_points,
                          nullptr,
                          reinterpret_cast<const Vec3f *>(points.data_ptr()),
                          num_points,
                          point_adjacency_size,
                          reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
                          reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
                          reinterpret_cast<Vec4h *>(adj_diff.data_ptr()));

    return adj_diff;
}

torch::Tensor trace_vorotracing_infer(torch::Tensor points_in,
                                      torch::Tensor diffuse_in,
                                      torch::Tensor specular_in,
                                      torch::Tensor density_in,
                                      torch::Tensor point_adjacency_in,
                                      torch::Tensor point_adjacency_offsets_in,
                                      torch::Tensor rays_in,
                                      torch::Tensor start_point_in,
                                      torch::Tensor adjacent_diff_in,
                                      torch::Tensor ray_perm_in,
                                      float weight_threshold,
                                      uint32_t max_intersections,
                                      float cell_skip_threshold)
{
    torch::Tensor points = points_in.contiguous();
    // Optional warp-coherence permutation: thread t processes ray ray_perm[t].
    // Empty tensor => identity (thread t -> ray t).
    torch::Tensor ray_perm = ray_perm_in.contiguous();
    const uint32_t *ray_perm_ptr =
        ray_perm.numel() ? reinterpret_cast<const uint32_t *>(ray_perm.data_ptr()) : nullptr;
    torch::Tensor diffuse = diffuse_in.contiguous();
    torch::Tensor specular = specular_in.contiguous();
    torch::Tensor density = density_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets = point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();
    torch::Tensor adjacent_diff = adjacent_diff_in.contiguous();

    validate_scene_data(points_in, diffuse_in, point_adjacency_in, point_adjacency_offsets_in);

    uint32_t num_points = points.size(0);
    uint32_t num_rays = rays.numel() / 6;

    ASSERT(diffuse.scalar_type() == torch::kFloat16, "inference kernel requires float16 diffuse");
    ASSERT(specular.scalar_type() == torch::kFloat16, "inference kernel requires float16 specular");
    ASSERT(density.scalar_type() == at::kFloat, "density must have float32 dtype");
    ASSERT(rays.scalar_type() == at::kFloat, "rays must have float32 dtype");
    ASSERT(start_point.scalar_type() == at::kUInt32, "start_point must have uint32 dtype");

    TraceSettings settings = default_trace_settings();
    settings.weight_threshold = weight_threshold;
    settings.max_intersections = max_intersections;
    settings.cell_skip_threshold = cell_skip_threshold;

    std::vector<int64_t> output_shape;
    for (int i = 0; i < rays.dim() - 1; i++)
    {
        output_shape.push_back(rays.size(i));
    }
    output_shape.push_back(4);
    torch::Tensor output_rgba =
        torch::empty(output_shape, torch::TensorOptions().dtype(torch::kFloat16).device(rays.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    at::cuda::setCurrentCUDAStream(stream);

    launch_kernel_1d<BLOCK_SIZE_OCTMAP>(trace_vorotracing_infer_kernel<OCTMAP_RES, OCTMAP_RES_SPEC>,
                                        num_rays,
                                        nullptr,
                                        settings,
                                        reinterpret_cast<const Vec3f *>(points.data_ptr()),
                                        reinterpret_cast<const __half *>(diffuse.data_ptr()),
                                        reinterpret_cast<const __half *>(specular.data_ptr()),
                                        reinterpret_cast<const float *>(density.data_ptr()),
                                        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
                                        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
                                        reinterpret_cast<const Vec4h *>(adjacent_diff.data_ptr()),
                                        reinterpret_cast<const Ray *>(rays.data_ptr()),
                                        num_rays,
                                        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
                                        reinterpret_cast<__half *>(output_rgba.data_ptr()),
                                        ray_perm_ptr);

    return output_rgba;
}

////////////////////////////////////////////////////////////
// Inference entry point (int8 quantized attributes)
////////////////////////////////////////////////////////////

template <int diff_map_res, int spec_map_res>
__global__ void trace_vorotracing_infer_q8_kernel(TraceSettings settings,
                                                  const Vec3f *__restrict__ points,
                                                  const uint8_t *__restrict__ diffuse,
                                                  const uint8_t *__restrict__ specular,
                                                  const float *__restrict__ density,
                                                  const uint32_t *__restrict__ point_adjacency,
                                                  const uint32_t *__restrict__ point_adjacency_offsets,
                                                  const Vec4h *__restrict__ adjacent_diff,
                                                  const Ray *__restrict__ rays,
                                                  uint32_t num_rays,
                                                  const uint32_t *__restrict__ start_point_index,
                                                  float diff_scale,
                                                  float diff_offset,
                                                  float spec_scale,
                                                  float spec_offset,
                                                  __half *__restrict__ ray_rgba)
{
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    uint32_t start_point = start_point_index[thread_idx];

    RayIntegratorOctMapInferQ8<diff_map_res, spec_map_res> ray_integrator(ray,
                                                                          diffuse,
                                                                          specular,
                                                                          density,
                                                                          settings.weight_threshold,
                                                                          diff_scale,
                                                                          diff_offset,
                                                                          spec_scale,
                                                                          spec_offset,
                                                                          settings.cell_skip_threshold);

    trace_octmap<4>(ray,
                    points,
                    point_adjacency,
                    point_adjacency_offsets,
                    adjacent_diff,
                    (float *)nullptr,
                    start_point,
                    settings.max_intersections,
                    ray_integrator);

    Vec3f accumulated_rgb = ray_integrator.get_accumulated_rgb();
    float transmittance = ray_integrator.get_transmittance();

    for (uint32_t i = 0; i < 3; ++i)
    {
        ray_rgba[thread_idx * 4 + i] = __float2half(accumulated_rgb[i]);
    }
    ray_rgba[thread_idx * 4 + 3] = __float2half(1 - transmittance);
}

torch::Tensor trace_vorotracing_infer_q8(torch::Tensor points_in,
                                         torch::Tensor diffuse_in,
                                         torch::Tensor specular_in,
                                         torch::Tensor density_in,
                                         torch::Tensor point_adjacency_in,
                                         torch::Tensor point_adjacency_offsets_in,
                                         torch::Tensor rays_in,
                                         torch::Tensor start_point_in,
                                         torch::Tensor adjacent_diff_in,
                                         float diff_scale,
                                         float diff_offset,
                                         float spec_scale,
                                         float spec_offset,
                                         float weight_threshold,
                                         uint32_t max_intersections,
                                         float cell_skip_threshold)
{
    torch::Tensor points = points_in.contiguous();
    torch::Tensor diffuse = diffuse_in.contiguous();
    torch::Tensor specular = specular_in.contiguous();
    torch::Tensor density = density_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets = point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();
    torch::Tensor adjacent_diff = adjacent_diff_in.contiguous();

    validate_scene_data(points_in, diffuse_in, point_adjacency_in, point_adjacency_offsets_in);

    uint32_t num_points = points.size(0);
    uint32_t num_rays = rays.numel() / 6;

    ASSERT(diffuse.scalar_type() == at::kByte, "q8 inference kernel requires uint8 diffuse");
    ASSERT(specular.scalar_type() == at::kByte, "q8 inference kernel requires uint8 specular");

    TraceSettings settings = default_trace_settings();
    settings.weight_threshold = weight_threshold;
    settings.max_intersections = max_intersections;
    settings.cell_skip_threshold = cell_skip_threshold;

    std::vector<int64_t> output_shape;
    for (int i = 0; i < rays.dim() - 1; i++)
    {
        output_shape.push_back(rays.size(i));
    }
    output_shape.push_back(4);
    torch::Tensor output_rgba =
        torch::empty(output_shape, torch::TensorOptions().dtype(torch::kFloat16).device(rays.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    at::cuda::setCurrentCUDAStream(stream);

    launch_kernel_1d<BLOCK_SIZE_OCTMAP>(trace_vorotracing_infer_q8_kernel<OCTMAP_RES, OCTMAP_RES_SPEC>,
                                        num_rays,
                                        nullptr,
                                        settings,
                                        reinterpret_cast<const Vec3f *>(points.data_ptr()),
                                        reinterpret_cast<const uint8_t *>(diffuse.data_ptr()),
                                        reinterpret_cast<const uint8_t *>(specular.data_ptr()),
                                        reinterpret_cast<const float *>(density.data_ptr()),
                                        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
                                        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
                                        reinterpret_cast<const Vec4h *>(adjacent_diff.data_ptr()),
                                        reinterpret_cast<const Ray *>(rays.data_ptr()),
                                        num_rays,
                                        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
                                        diff_scale,
                                        diff_offset,
                                        spec_scale,
                                        spec_offset,
                                        reinterpret_cast<__half *>(output_rgba.data_ptr()));

    return output_rgba;
}

////////////////////////////////////////////////////////////
// Backward entry point
////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trace_vorotracing_bwd(torch::Tensor points_in,
                      torch::Tensor diffuse_in,
                      torch::Tensor specular_in,
                      torch::Tensor density_in,
                      torch::Tensor point_adjacency_in,
                      torch::Tensor point_adjacency_offsets_in,
                      torch::Tensor rays_in,
                      torch::Tensor start_point_in,
                      torch::Tensor rgb_out,
                      torch::Tensor rgb_grad_in,
                      std::optional<torch::Tensor> depth_quantiles_in,
                      std::optional<torch::Tensor> depth_indices_in,
                      std::optional<torch::Tensor> depth_grad_in,
                      std::optional<torch::Tensor> ray_error_in,
                      float contribution_alpha_grad,
                      torch::Tensor distortion_in,
                      std::optional<torch::Tensor> distortion_grad_in,
                      torch::Tensor W_total_in,
                      torch::Tensor S_total_in,
                      torch::Tensor t_far_in,
                      float weight_threshold,
                      uint32_t max_intersections)
{
    torch::Tensor points = points_in.contiguous();
    torch::Tensor diffuse = diffuse_in.contiguous();
    torch::Tensor specular = specular_in.contiguous();
    torch::Tensor density = density_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets = point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();

    validate_scene_data(points_in, diffuse_in, point_adjacency_in, point_adjacency_offsets_in);

    bool return_depth = depth_quantiles_in.has_value();
    bool return_error = ray_error_in.has_value();

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);
    uint32_t num_rays = rays.numel() / 6;
    uint32_t num_depth_quantiles = 0;

    ASSERT(diffuse.scalar_type() == specular.scalar_type(), "diffuse and specular must have the same dtype");
    ASSERT(diffuse.size(0) == specular.size(0) && diffuse.size(0) == density.size(0) &&
               diffuse.size(0) == (int64_t)num_points,
           "diffuse, specular, density must each have num_points rows");
    ASSERT(density.scalar_type() == at::kFloat, "density must have float32 dtype");

    ASSERT(rays.size(-1) == 6, "rays must have 6 as the last dimension");
    ASSERT(rays.scalar_type() == at::kFloat, "rays must have float32 dtype");
    ASSERT(rays.device().type() == at::kCUDA, "rays must be on CUDA device");

    ASSERT(start_point.numel() == num_rays, "start_point must have the same batch size as rays");
    ASSERT(start_point.scalar_type() == at::kUInt32, "start_point must have uint32 dtype");
    ASSERT(start_point.device().type() == at::kCUDA, "start_point must be on CUDA device");

    torch::Tensor rgb_grad_in_c = rgb_grad_in.contiguous();
    ASSERT(rgb_grad_in_c.size(-1) == 4, "rgb_grad_in must have 4 as the last dimension");
    ASSERT(rgb_grad_in_c.scalar_type() == diffuse.scalar_type(),
           "rgb_grad_in must have the same dtype as diffuse/specular");
    ASSERT(rgb_grad_in_c.device().type() == at::kCUDA, "rgb_grad_in must be on CUDA device");
    ASSERT(rgb_grad_in_c.numel() / 4 == num_rays, "rgb_grad_in must have the same batch size as rays");

    torch::Tensor depth_quantiles;
    torch::Tensor depth_indices;
    torch::Tensor depth_grad;
    if (return_depth)
    {
        depth_quantiles = depth_quantiles_in.value().contiguous();
        num_depth_quantiles = depth_quantiles.size(-1);

        ASSERT(depth_quantiles.scalar_type() == at::kFloat, "depth_quantiles must have float32 dtype");
        ASSERT(depth_quantiles.device().type() == at::kCUDA, "depth_quantiles must be on CUDA device");
        ASSERT(depth_quantiles.numel() == num_rays * num_depth_quantiles,
               "depth_quantiles must have the same batch size as rays");

        ASSERT(depth_grad_in.has_value(), "depth_grad must be provided if depth_quantiles is provided");

        depth_indices = depth_indices_in.value().contiguous();

        ASSERT(depth_indices.scalar_type() == at::kUInt32, "depth_indices must have uint32 dtype");
        ASSERT(depth_indices.device().type() == at::kCUDA, "depth_indices must be on CUDA device");
        ASSERT(depth_indices.numel() == num_rays * num_depth_quantiles,
               "depth_indices must have the same batch size as rays");

        depth_grad = depth_grad_in.value().contiguous();

        ASSERT(depth_grad.size(-1) == num_depth_quantiles,
               "depth_grad must have the same number of depth quantiles as depth_quantiles");
        ASSERT(depth_grad.scalar_type() == at::kFloat, "depth_grad must have float32 dtype");
        ASSERT(depth_grad.device().type() == at::kCUDA, "depth_grad must be on CUDA device");
        ASSERT(depth_grad.numel() == num_rays * num_depth_quantiles,
               "depth_grad must have the same batch size as rays");
    }

    torch::Tensor ray_error;
    torch::Tensor point_error;
    if (return_error)
    {
        ray_error = ray_error_in.value().contiguous();

        ASSERT(ray_error.scalar_type() == diffuse.scalar_type(),
               "ray_error must have the same dtype as diffuse/specular");
        ASSERT(ray_error.device().type() == at::kCUDA, "ray_error must be on CUDA device");
        ASSERT(ray_error.numel() == num_rays, "ray_error must have the same batch size as rays");

        point_error =
            torch::zeros({num_points, 1}, torch::TensorOptions().dtype(diffuse.scalar_type()).device(rays.device()));
    }

    torch::Tensor W_total = W_total_in.contiguous();
    torch::Tensor S_total = S_total_in.contiguous();
    torch::Tensor t_far = t_far_in.contiguous();
    torch::Tensor distortion = distortion_in.contiguous();
    ASSERT(W_total.scalar_type() == at::kFloat, "W_total must have float32 dtype");
    ASSERT(S_total.scalar_type() == at::kFloat, "S_total must have float32 dtype");
    ASSERT(t_far.scalar_type() == at::kFloat, "t_far must have float32 dtype");
    ASSERT(distortion.scalar_type() == at::kFloat, "distortion must have float32 dtype");
    ASSERT(W_total.numel() == num_rays, "W_total must have the same batch size as rays");
    ASSERT(S_total.numel() == num_rays, "S_total must have the same batch size as rays");
    ASSERT(t_far.numel() == num_rays, "t_far must have the same batch size as rays");
    ASSERT(distortion.numel() == num_rays, "distortion must have the same batch size as rays");

    torch::Tensor distortion_grad;
    bool has_distortion_grad = distortion_grad_in.has_value();
    if (has_distortion_grad)
    {
        distortion_grad = distortion_grad_in.value().contiguous();
        ASSERT(distortion_grad.scalar_type() == at::kFloat, "distortion_grad must have float32 dtype");
        ASSERT(distortion_grad.numel() == num_rays, "distortion_grad must have the same batch size as rays");
    }

    TraceSettings settings = default_trace_settings();
    settings.weight_threshold = weight_threshold;
    settings.max_intersections = max_intersections;

    auto color_options = torch::TensorOptions().dtype(diffuse.scalar_type()).device(rays.device());
    auto density_options = torch::TensorOptions().dtype(torch::kFloat32).device(rays.device());

    torch::Tensor diffuse_grad = torch::zeros_like(diffuse, color_options);
    torch::Tensor specular_grad = torch::zeros_like(specular, color_options);
    torch::Tensor density_grad = torch::zeros_like(density, density_options);

    std::vector<int64_t> points_grad_shape = {(int64_t)num_points, 3};
    torch::Tensor points_grad =
        torch::zeros(points_grad_shape, torch::TensorOptions().dtype(rays.scalar_type()).device(rays.device()));

    torch::Tensor ray_grad = torch::empty_like(rays);

    set_default_stream();

    CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
    launch_kernel_1d<256>(prefetch_adjacent_diff_octmap_kernel,
                          num_points,
                          nullptr,
                          reinterpret_cast<const Vec3f *>(points.data_ptr()),
                          num_points,
                          point_adjacency_size,
                          reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
                          reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
                          adjacent_diff.begin());

    if (diffuse.scalar_type() == torch::kFloat32)
    {
        launch_kernel_1d<BLOCK_SIZE_OCTMAP>(
            trace_vorotracing_bwd_kernel<float, OCTMAP_RES, OCTMAP_RES_SPEC>,
            num_rays,
            nullptr,
            settings,
            reinterpret_cast<const Vec3f *>(points.data_ptr()),
            reinterpret_cast<const float *>(diffuse.data_ptr()),
            reinterpret_cast<const float *>(specular.data_ptr()),
            reinterpret_cast<const float *>(density.data_ptr()),
            reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
            reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
            adjacent_diff.begin(),
            reinterpret_cast<const Ray *>(rays.data_ptr()),
            num_rays,
            reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
            num_depth_quantiles,
            return_depth ? reinterpret_cast<const float *>(depth_quantiles.data_ptr()) : nullptr,
            return_depth ? reinterpret_cast<const uint32_t *>(depth_indices.data_ptr()) : nullptr,
            reinterpret_cast<const float *>(rgb_out.data_ptr()),
            reinterpret_cast<const float *>(rgb_grad_in_c.data_ptr()),
            return_depth ? reinterpret_cast<const float *>(depth_grad.data_ptr()) : nullptr,
            return_error ? reinterpret_cast<const float *>(ray_error.data_ptr()) : nullptr,
            contribution_alpha_grad,
            reinterpret_cast<const float *>(distortion.data_ptr()),
            has_distortion_grad ? reinterpret_cast<const float *>(distortion_grad.data_ptr()) : nullptr,
            reinterpret_cast<const float *>(W_total.data_ptr()),
            reinterpret_cast<const float *>(S_total.data_ptr()),
            reinterpret_cast<const float *>(t_far.data_ptr()),
            reinterpret_cast<Ray *>(ray_grad.data_ptr()),
            reinterpret_cast<Vec3f *>(points_grad.data_ptr()),
            reinterpret_cast<float *>(diffuse_grad.data_ptr()),
            reinterpret_cast<float *>(specular_grad.data_ptr()),
            reinterpret_cast<float *>(density_grad.data_ptr()),
            return_error ? reinterpret_cast<float *>(point_error.data_ptr()) : nullptr);
    }
    else if (diffuse.scalar_type() == torch::kFloat16)
    {
        launch_kernel_1d<BLOCK_SIZE_OCTMAP>(
            trace_vorotracing_bwd_kernel<__half, OCTMAP_RES, OCTMAP_RES_SPEC>,
            num_rays,
            nullptr,
            settings,
            reinterpret_cast<const Vec3f *>(points.data_ptr()),
            reinterpret_cast<const __half *>(diffuse.data_ptr()),
            reinterpret_cast<const __half *>(specular.data_ptr()),
            reinterpret_cast<const float *>(density.data_ptr()),
            reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
            reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
            adjacent_diff.begin(),
            reinterpret_cast<const Ray *>(rays.data_ptr()),
            num_rays,
            reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
            num_depth_quantiles,
            return_depth ? reinterpret_cast<const float *>(depth_quantiles.data_ptr()) : nullptr,
            return_depth ? reinterpret_cast<const uint32_t *>(depth_indices.data_ptr()) : nullptr,
            reinterpret_cast<const __half *>(rgb_out.data_ptr()),
            reinterpret_cast<const __half *>(rgb_grad_in_c.data_ptr()),
            return_depth ? reinterpret_cast<const float *>(depth_grad.data_ptr()) : nullptr,
            return_error ? reinterpret_cast<const __half *>(ray_error.data_ptr()) : nullptr,
            contribution_alpha_grad,
            reinterpret_cast<const float *>(distortion.data_ptr()),
            has_distortion_grad ? reinterpret_cast<const float *>(distortion_grad.data_ptr()) : nullptr,
            reinterpret_cast<const float *>(W_total.data_ptr()),
            reinterpret_cast<const float *>(S_total.data_ptr()),
            reinterpret_cast<const float *>(t_far.data_ptr()),
            reinterpret_cast<Ray *>(ray_grad.data_ptr()),
            reinterpret_cast<Vec3f *>(points_grad.data_ptr()),
            reinterpret_cast<__half *>(diffuse_grad.data_ptr()),
            reinterpret_cast<__half *>(specular_grad.data_ptr()),
            reinterpret_cast<float *>(density_grad.data_ptr()),
            return_error ? reinterpret_cast<__half *>(point_error.data_ptr()) : nullptr);
    }
    else
    {
        throw std::runtime_error("Unsupported color dtype. Only float32 and float16 are supported.");
    }

    return std::make_tuple(points_grad, diffuse_grad, specular_grad, density_grad, ray_grad, point_error);
}

} // namespace vorotracing
