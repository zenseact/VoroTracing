#pragma once

#include <float.h>
#include <stdint.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace vorotracing
{

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
                      bool return_contribution);

torch::Tensor
trace_vorotracing_prefetch_adj(torch::Tensor points_in,
                               torch::Tensor point_adjacency_in,
                               torch::Tensor point_adjacency_offsets_in);

torch::Tensor
trace_vorotracing_infer(torch::Tensor points_in,
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
                        float cell_skip_threshold);

torch::Tensor
trace_vorotracing_infer_q8(torch::Tensor points_in,
                           torch::Tensor diffuse_in,
                           torch::Tensor specular_in,
                           torch::Tensor density_in,
                           torch::Tensor point_adjacency_in,
                           torch::Tensor point_adjacency_offsets_in,
                           torch::Tensor rays_in,
                           torch::Tensor start_point_in,
                           torch::Tensor adjacent_diff_in,
                           float diff_scale, float diff_offset,
                           float spec_scale, float spec_offset,
                           float weight_threshold,
                           uint32_t max_intersections,
                           float cell_skip_threshold);

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
                      uint32_t max_intersections);

} // namespace vorotracing
