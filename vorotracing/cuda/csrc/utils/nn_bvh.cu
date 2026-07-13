#include "nn_bvh.h"

// Define implementation for cuBQL builder - MUST come before any cuBQL include
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include <cuBQL/bvh.h>

#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/pointData/findClosest.h"
#include <torch/torch.h>

// Kernel to create boxes from float3 points
static __global__ void create_boxes_kernel(cuBQL::box_t<float, 3> *boxes, const float3 *points, int numPoints)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numPoints)
        return;

    float3 p = points[gid];
    cuBQL::vec_t<float, 3> pos;
    pos.x = p.x;
    pos.y = p.y;
    pos.z = p.z;

    boxes[gid].lower = pos;
    boxes[gid].upper = pos;
}

// Kernel for findClosest
static __global__ void find_closest_kernel(int *indices,
                                           const float3 *queries,
                                           int numQueries,
                                           cuBQL::BinaryBVH<float, 3> bvh,
                                           const float3 *points)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numQueries)
        return;

    float3 q = queries[gid];
    indices[gid] = cuBQL::points::findClosest(bvh, points, q);
}

NNBVH::NNBVH()
{
    auto bvh = new cuBQL::BinaryBVH<float, 3>();
    bvh->nodes = nullptr;
    bvh->primIDs = nullptr;
    bvh->numNodes = 0;
    bvh->numPrims = 0;
    bvh_ptr = (void *)bvh;
}

NNBVH::~NNBVH()
{
    auto bvh = (cuBQL::BinaryBVH<float, 3> *)bvh_ptr;
    if (bvh->nodes || bvh->primIDs)
    {
        cuBQL::cuda::free(*bvh);
    }
    delete bvh;
}

void NNBVH::build(torch::Tensor points)
{
    TORCH_CHECK(points.device().is_cuda(), "points must be on CUDA");
    int num_prims = points.size(0);
    float3 *d_points = (float3 *)points.data_ptr<float>();

    auto bvh = (cuBQL::BinaryBVH<float, 3> *)bvh_ptr;

    // Free old BVH if it exists
    if (bvh->nodes || bvh->primIDs)
    {
        cuBQL::cuda::free(*bvh);
    }

    cuBQL::box_t<float, 3> *d_boxes;
    cudaMalloc(&d_boxes, num_prims * sizeof(cuBQL::box_t<float, 3>));

    int blockSize = 256;
    int numBlocksPoints = (num_prims + blockSize - 1) / blockSize;
    create_boxes_kernel<<<numBlocksPoints, blockSize>>>(d_boxes, d_points, num_prims);

    cuBQL::BuildConfig buildConfig;
    cuBQL::gpuBuilder(*bvh, d_boxes, num_prims, buildConfig);

    cudaFree(d_boxes);
    num_points = num_prims;
}

torch::Tensor NNBVH::query(torch::Tensor points, torch::Tensor queries)
{
    auto bvh = (cuBQL::BinaryBVH<float, 3> *)bvh_ptr;
    TORCH_CHECK(bvh->nodes != nullptr, "BVH must be built before querying");
    TORCH_CHECK(queries.device().is_cuda(), "queries must be on CUDA");

    int num_queries = queries.numel() / 3;
    float3 *d_queries = (float3 *)queries.data_ptr<float>();
    float3 *d_points = (float3 *)points.data_ptr<float>();

    auto result = torch::empty({num_queries}, torch::dtype(torch::kInt32).device(queries.device()));
    int *d_indices = result.data_ptr<int>();

    int blockSize = 256;
    int numBlocksQueries = (num_queries + blockSize - 1) / blockSize;
    find_closest_kernel<<<numBlocksQueries, blockSize>>>(d_indices, d_queries, num_queries, *bvh, d_points);

    return result;
}

// Keep the old function for backward compatibility or simple one-off use
torch::Tensor nearest_neighbor_bvh(torch::Tensor points, torch::Tensor queries)
{
    NNBVH bvh;
    bvh.build(points);
    return bvh.query(points, queries);
}
