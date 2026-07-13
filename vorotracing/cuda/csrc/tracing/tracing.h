#pragma once

#include <float.h>
#include <stdint.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace vorotracing
{

struct TraceSettings
{
    float weight_threshold;
    uint32_t max_intersections;
    // Per-cell contribution gate (inference only): if
    // transmittance * alpha < cell_skip_threshold, skip the diffuse/specular
    // texture loads for this cell. Set to 0.0 to disable. Saves the bulk of
    // per-cell L2 traffic for cells whose color contribution would be tiny,
    // at the cost of a small extra branch + bounded color error.
    float cell_skip_threshold;
};

inline TraceSettings default_trace_settings()
{
    TraceSettings settings;
    settings.weight_threshold = 0.001f;
    settings.max_intersections = 1024;
    settings.cell_skip_threshold = 0.0f;
    return settings;
}

} // namespace vorotracing
