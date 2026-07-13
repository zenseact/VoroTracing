#pragma once

#include <torch/torch.h>
#include <tuple>

namespace vorotracing
{

std::tuple<torch::Tensor, torch::Tensor>
farthest_neighbor(torch::Tensor points_in,
                  torch::Tensor point_adjacency_in,
                  torch::Tensor point_adjacency_offsets_in);

} // namespace vorotracing
