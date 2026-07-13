#include <torch/extension.h>

#include "tracing/tracing_octmap.h"
#include "utils/cuda_helpers.h"
#include "utils/farthest_neighbor.h"
#include "utils/nn_bvh.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    vorotracing::global_cuda_init();

    m.def("trace_vorotracing_fwd",
          &vorotracing::trace_vorotracing_fwd,
          "VoroTracing octahedral-map forward");
    m.def("trace_vorotracing_bwd",
          &vorotracing::trace_vorotracing_bwd,
          "VoroTracing octahedral-map backward");
    m.def("trace_vorotracing_prefetch_adj",
          &vorotracing::trace_vorotracing_prefetch_adj,
          "Prefetch adjacent differences for VoroTracing inference");
    m.def("trace_vorotracing_infer",
          &vorotracing::trace_vorotracing_infer,
          "VoroTracing octahedral-map inference");
    m.def("trace_vorotracing_infer_q8",
          &vorotracing::trace_vorotracing_infer_q8,
          "VoroTracing octahedral-map inference with int8 attributes");

    m.def("farthest_neighbor",
          &vorotracing::farthest_neighbor,
          "Find the farthest Voronoi neighbor and mean cell radius");

    py::class_<NNBVH>(m, "NNBVH")
        .def(py::init<>())
        .def("build", &NNBVH::build, py::arg("points"))
        .def("query", &NNBVH::query, py::arg("points"), py::arg("queries"));

    m.def("nearest_neighbor_bvh",
          &nearest_neighbor_bvh,
          "Find nearest neighbors using a temporary cuBQL BVH");
}
