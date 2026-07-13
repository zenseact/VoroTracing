#pragma once
#include <torch/torch.h>

class NNBVH
{
  public:
    NNBVH();
    ~NNBVH();

    void build(torch::Tensor points);
    torch::Tensor query(torch::Tensor points, torch::Tensor queries);

  private:
    void *bvh_ptr; // Store cuBQL::BinaryBVH<float, 3>*
    int num_points = 0;
};

torch::Tensor nearest_neighbor_bvh(torch::Tensor points, torch::Tensor queries);