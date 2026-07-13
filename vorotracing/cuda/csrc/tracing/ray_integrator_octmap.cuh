#pragma once

#include <cuda_fp16.h>

#include "../utils/geometry.h"
#include "../utils/typing.cuh"
#include <assert.h>
#include <cuda_runtime.h>

namespace vorotracing
{

// Octahedral encoding and wrap-indexing helpers shared by all four integrator
// classes (training fwd/bwd, fp16 infer, q8 infer). Templated on the texture
// resolution `R` so that diffuse and specular maps can carry different
// resolutions per integrator instance.
template <int R> __device__ inline Vec2f oct_encode(const Vec3f &dir)
{
    float l1 = fabsf(dir[0]) + fabsf(dir[1]) + fabsf(dir[2]);
    float inv_l1 = 1.0f / fmaxf(l1, 1e-10f);
    float u = dir[0] * inv_l1;
    float v = dir[1] * inv_l1;

    if (dir[2] < 0.0f)
    {
        float ou = (1.0f - fabsf(v)) * copysignf(1.0f, u);
        float ov = (1.0f - fabsf(u)) * copysignf(1.0f, v);
        u = ou;
        v = ov;
    }

    float tex_u = (u * 0.5f + 0.5f) * R - 0.5f;
    float tex_v = (v * 0.5f + 0.5f) * R - 0.5f;
    return Vec2f(tex_u, tex_v);
}

template <int R> __device__ inline int oct_wrap_index(int u_texel, int v_texel)
{
    if (u_texel >= 0 && u_texel < R && v_texel >= 0 && v_texel < R)
        return v_texel * R + u_texel;

    int wu = ((u_texel % R) + R) % R;
    int wv = ((v_texel % R) + R) % R;

    int fold_u = abs(u_texel / R) + (u_texel < 0 ? 1 : 0);
    int fold_v = abs(v_texel / R) + (v_texel < 0 ? 1 : 0);

    if (((fold_u ^ fold_v) & 1) != 0)
    {
        wu = R - 1 - wu;
        wv = R - 1 - wv;
    }

    return wv * R + wu;
}

template <typename attr_scalar, int diff_map_res, int spec_map_res> class RayIntegratorOctMap
{
  public:
    __device__ RayIntegratorOctMap(Ray ray,
                                   const attr_scalar *__restrict__ diffuse,
                                   const attr_scalar *__restrict__ specular,
                                   const float *__restrict__ density,
                                   float weight_threshold,
                                   const float *quantile_thresholds,
                                   uint32_t num_depth_quantiles,
                                   float *quantile_depths,
                                   uint32_t *quantile_point_indices)
        : ray_(ray), diffuse_(diffuse), specular_(specular), density_(density), weight_threshold_(weight_threshold),
          quantile_thresholds_(quantile_thresholds), num_depth_quantiles_(num_depth_quantiles),
          quantile_depths_(quantile_depths), quantile_point_indices_(quantile_point_indices)
    {
        assert(num_depth_quantiles_ == 0 ||
               (quantile_thresholds_ != nullptr && quantile_depths_ != nullptr && quantile_point_indices_ != nullptr));

        if (num_depth_quantiles_ > 0)
        {
            current_quantile_value_ = quantile_thresholds_[0];
        }

        init_specular_bilinear(oct_encode<spec_map_res>(-ray.direction));
    }

    __device__ Vec3f get_accumulated_rgb() const { return accumulated_rgb_; }

    __device__ float get_transmittance() const { return transmittance_; }

    __device__ uint32_t get_num_filled_quantiles() const { return current_quantile_idx_; }

    // mip-NeRF 360 distortion loss accumulators (eq. 15, Barron et al. 2022).
    // Computed in disparity-style s-space (s = 1 - 1/(1+t)), so s in [0, 1) for
    // t in [0, ∞). No per-ray length normalization — t_far_ kept only for the
    // ABI, not used in the loss value.
    __device__ float get_W_total() const { return W_run_; }
    __device__ float get_S_total() const { return S_run_; }
    __device__ float get_t_far() const { return t_far_; }
    __device__ float get_distortion() const
    {
        return 2.0f * dist_accum_ + self_accum_ / 3.0f;
    }

    __device__ bool integrate_cell(uint32_t point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point,
                                   const Vec3f &next_point,
                                   attr_scalar *point_contribution)
    {
        Vec3f rgb;
        float s;

        load_attributes(point_idx, t_0, t_1, current_point, rgb, s);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s * delta_t);
        float weight = transmittance_ * alpha;

        // Distortion loss accumulators in disparity-style s-space: s = 1 - 1/(1+t).
        // Bounded in [0, 1) for any t ≥ 0, so no per-ray normalization needed.
        // Near-camera cells get larger δ_s than far cells of the same Δt — correct bias.
        float s_0 = 1.0f - 1.0f / (1.0f + t_0);
        float s_1 = 1.0f - 1.0f / (1.0f + t_1);
        float mid_s = 0.5f * (s_0 + s_1);
        float delta_s = s_1 - s_0;
        dist_accum_ += weight * (mid_s * W_run_ - S_run_);
        self_accum_ += weight * weight * delta_s;
        W_run_ += weight;
        S_run_ += weight * mid_s;
        if (t_1 > t_far_) t_far_ = t_1;

        if (point_contribution)
        {
            atomicAdd(point_contribution + point_idx, from_float<attr_scalar>(weight));
        }
        accumulated_rgb_ += weight * rgb;

        float next_transmittance = transmittance_ * (1 - alpha);
        while (current_quantile_idx_ < num_depth_quantiles_ && next_transmittance < current_quantile_value_)
        {
            quantile_depths_[current_quantile_idx_] = t_0 + logf(transmittance_ / current_quantile_value_) / s;
            quantile_point_indices_[current_quantile_idx_] = point_idx;
            current_quantile_idx_++;
            if (current_quantile_idx_ < num_depth_quantiles_)
            {
                current_quantile_value_ = quantile_thresholds_[current_quantile_idx_];
            }
        }

        transmittance_ = next_transmittance;

        return transmittance_ > weight_threshold_;
    }

  private:
    Vec3f accumulated_rgb_ = Vec3f::Zero();
    float transmittance_ = 1.0f;
    Ray ray_;
    const attr_scalar *__restrict__ diffuse_ = nullptr;
    const attr_scalar *__restrict__ specular_ = nullptr;
    const float *__restrict__ density_ = nullptr;

    const float *quantile_thresholds_ = nullptr;
    uint32_t num_depth_quantiles_ = 0;
    float *quantile_depths_ = nullptr;
    uint32_t *quantile_point_indices_ = nullptr;

    uint32_t current_quantile_idx_ = 0;
    float current_quantile_value_ = 0.0f;

    float weight_threshold_ = 1e-6f;

    // Distortion loss accumulators (mip-NeRF 360 eq. 15).
    float W_run_ = 0.0f;
    float S_run_ = 0.0f;
    float dist_accum_ = 0.0f;
    float self_accum_ = 0.0f;
    float t_far_ = 0.0f;

    int spec_idx00_, spec_idx10_, spec_idx01_, spec_idx11_;
    float spec_fu_, spec_fv_;

    static constexpr int single_map_size_diff_ = 3 * diff_map_res * diff_map_res;
    static constexpr int single_map_size_spec_ = 3 * spec_map_res * spec_map_res;

    __device__ void init_specular_bilinear(Vec2f uv)
    {
        float u = uv[0];
        float v = uv[1];
        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);

        spec_idx00_ = oct_wrap_index<spec_map_res>(u0, v0);
        spec_idx10_ = oct_wrap_index<spec_map_res>(u0 + 1, v0);
        spec_idx01_ = oct_wrap_index<spec_map_res>(u0, v0 + 1);
        spec_idx11_ = oct_wrap_index<spec_map_res>(u0 + 1, v0 + 1);
        spec_fu_ = u - u0;
        spec_fv_ = v - v0;
    }

    __device__ Vec3f bilinear_lookup(const attr_scalar *map_ptr, Vec2f uv)
    {
        float u = uv[0];
        float v = uv[1];

        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);
        int u1 = u0 + 1;
        int v1 = v0 + 1;
        float fu = u - u0;
        float fv = v - v0;

        int idx00 = oct_wrap_index<diff_map_res>(u0, v0);
        int idx10 = oct_wrap_index<diff_map_res>(u1, v0);
        int idx01 = oct_wrap_index<diff_map_res>(u0, v1);
        int idx11 = oct_wrap_index<diff_map_res>(u1, v1);

        float w00 = (1.0f - fu) * (1.0f - fv);
        float w10 = fu * (1.0f - fv);
        float w01 = (1.0f - fu) * fv;
        float w11 = fu * fv;

        Vec3f rgb = Vec3f::Zero();
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            float val = w00 * to_float(map_ptr[idx00 * 3 + c]) + w10 * to_float(map_ptr[idx10 * 3 + c]) +
                        w01 * to_float(map_ptr[idx01 * 3 + c]) + w11 * to_float(map_ptr[idx11 * 3 + c]);
            rgb[c] = val;
        }
        return rgb;
    }

    __device__ Vec3f specular_bilinear_lookup(const attr_scalar *map_ptr)
    {
        float w00 = (1.0f - spec_fu_) * (1.0f - spec_fv_);
        float w10 = spec_fu_ * (1.0f - spec_fv_);
        float w01 = (1.0f - spec_fu_) * spec_fv_;
        float w11 = spec_fu_ * spec_fv_;

        Vec3f rgb = Vec3f::Zero();
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            float val = w00 * to_float(map_ptr[spec_idx00_ * 3 + c]) + w10 * to_float(map_ptr[spec_idx10_ * 3 + c]) +
                        w01 * to_float(map_ptr[spec_idx01_ * 3 + c]) + w11 * to_float(map_ptr[spec_idx11_ * 3 + c]);
            rgb[c] = val;
        }
        return rgb;
    }

    __device__ void
    load_attributes(uint32_t point_idx, float t_0, float t_1, const Vec3f &current_point, Vec3f &rgb, float &s)
    {
        s = density_[point_idx];
        if (s > 1e-6f)
        {
            // Probe the diffuse octmap at the entry-face surface point (t_0)
            Vec3f hit = ray_.origin + t_0 * ray_.direction;
            Vec3f d = hit - current_point;
            float len = d.norm();
            if (len > 1e-8f)
                d /= len;
            else
                d = ray_.direction;

            const attr_scalar *diff_ptr = diffuse_ + point_idx * single_map_size_diff_;
            const attr_scalar *spec_ptr = specular_ + point_idx * single_map_size_spec_;

            Vec2f uv_diff = oct_encode<diff_map_res>(d);
            Vec3f diffuse_raw = bilinear_lookup(diff_ptr, uv_diff);
            Vec3f specular_raw = specular_bilinear_lookup(spec_ptr);

            Vec3f rgb_logit = diffuse_raw + specular_raw;
            for (int c = 0; c < 3; c++)
                rgb[c] = 1.0f / (1.0f + expf(-rgb_logit[c]));
        }
        else
        {
            rgb = Vec3f::Zero();
        }
    }
};

// Padded 4-channel half texel for vectorized 8-byte loads (LDG.E.64).
// Storage layout for fp16 inference: (num_points, R*R*4) per map, where R is
// the per-map resolution (diffuse and specular may differ). Packs the 3 active
// channels + 1 padding per texel. The padding is
// accepted because the headline bottleneck is L2 bandwidth wasted on
// uncoalesced sectors, not DRAM bytes.
struct __align__(8) Half4Texel
{
    __half c0, c1, c2, pad;
};

template <typename attr_scalar, int diff_map_res, int spec_map_res> class RayIntegratorOctMapInfer
{
  public:
    __device__ RayIntegratorOctMapInfer(Ray ray,
                                        const attr_scalar *__restrict__ diffuse,
                                        const attr_scalar *__restrict__ specular,
                                        const float *__restrict__ density,
                                        float weight_threshold,
                                        float cell_skip_threshold)
        : ray_(ray), diffuse_(reinterpret_cast<const Half4Texel *>(diffuse)),
          specular_(reinterpret_cast<const Half4Texel *>(specular)), density_(density),
          weight_threshold_(weight_threshold), cell_skip_threshold_(cell_skip_threshold)
    {
        init_specular_bilinear(oct_encode<spec_map_res>(-ray.direction));
    }

    __device__ Vec3f get_accumulated_rgb() const { return accumulated_rgb_; }

    __device__ float get_transmittance() const { return transmittance_; }

    __device__ bool integrate_cell(uint32_t point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point,
                                   const Vec3f &next_point,
                                   float * /* unused */)
    {
        // Cheap density gate first (1 fp32 load).
        float s = density_[point_idx];
        if (s <= 1e-6f)
        {
            return transmittance_ > weight_threshold_;
        }

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1.0f - expf(-s * delta_t);
        float weight = transmittance_ * alpha;

        // Skip the expensive texture loads + sigmoid when this cell's color
        // contribution would be smaller than cell_skip_threshold_. Still
        // advance transmittance so downstream cells see the correct occlusion.
        if (weight >= cell_skip_threshold_)
        {
            Vec3f rgb = compute_color(point_idx, t_0, t_1, current_point);
            accumulated_rgb_ += weight * rgb;
        }
        transmittance_ *= (1.0f - alpha);

        return transmittance_ > weight_threshold_;
    }

  private:
    Vec3f accumulated_rgb_ = Vec3f::Zero();
    float transmittance_ = 1.0f;
    Ray ray_;
    const Half4Texel *__restrict__ diffuse_ = nullptr;
    const Half4Texel *__restrict__ specular_ = nullptr;
    const float *__restrict__ density_ = nullptr;

    float weight_threshold_ = 1e-6f;
    float cell_skip_threshold_ = 0.0f;

    int spec_idx00_, spec_idx10_, spec_idx01_, spec_idx11_;
    float spec_fu_, spec_fv_;

    // Texel count per map (4-channel-packed layout: stride = R*R texels of 8
    // bytes each). Diffuse and specular can have different resolutions.
    static constexpr int single_map_size_diff_ = diff_map_res * diff_map_res;
    static constexpr int single_map_size_spec_ = spec_map_res * spec_map_res;

    __device__ void init_specular_bilinear(Vec2f uv)
    {
        float u = uv[0];
        float v = uv[1];
        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);

        spec_idx00_ = oct_wrap_index<spec_map_res>(u0, v0);
        spec_idx10_ = oct_wrap_index<spec_map_res>(u0 + 1, v0);
        spec_idx01_ = oct_wrap_index<spec_map_res>(u0, v0 + 1);
        spec_idx11_ = oct_wrap_index<spec_map_res>(u0 + 1, v0 + 1);
        spec_fu_ = u - u0;
        spec_fv_ = v - v0;
    }

    __device__ Vec3f bilinear_lookup(const Half4Texel *map_ptr, Vec2f uv)
    {
        float u = uv[0];
        float v = uv[1];

        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);
        int u1 = u0 + 1;
        int v1 = v0 + 1;
        float fu = u - u0;
        float fv = v - v0;

        int idx00 = oct_wrap_index<diff_map_res>(u0, v0);
        int idx10 = oct_wrap_index<diff_map_res>(u1, v0);
        int idx01 = oct_wrap_index<diff_map_res>(u0, v1);
        int idx11 = oct_wrap_index<diff_map_res>(u1, v1);

        float w00 = (1.0f - fu) * (1.0f - fv);
        float w10 = fu * (1.0f - fv);
        float w01 = (1.0f - fu) * fv;
        float w11 = fu * fv;

        Half4Texel t00 = map_ptr[idx00];
        Half4Texel t10 = map_ptr[idx10];
        Half4Texel t01 = map_ptr[idx01];
        Half4Texel t11 = map_ptr[idx11];

        Vec3f rgb;
        rgb[0] = w00 * to_float(t00.c0) + w10 * to_float(t10.c0) + w01 * to_float(t01.c0) + w11 * to_float(t11.c0);
        rgb[1] = w00 * to_float(t00.c1) + w10 * to_float(t10.c1) + w01 * to_float(t01.c1) + w11 * to_float(t11.c1);
        rgb[2] = w00 * to_float(t00.c2) + w10 * to_float(t10.c2) + w01 * to_float(t01.c2) + w11 * to_float(t11.c2);
        return rgb;
    }

    __device__ Vec3f specular_bilinear_lookup(const Half4Texel *map_ptr)
    {
        float w00 = (1.0f - spec_fu_) * (1.0f - spec_fv_);
        float w10 = spec_fu_ * (1.0f - spec_fv_);
        float w01 = (1.0f - spec_fu_) * spec_fv_;
        float w11 = spec_fu_ * spec_fv_;

        Half4Texel t00 = map_ptr[spec_idx00_];
        Half4Texel t10 = map_ptr[spec_idx10_];
        Half4Texel t01 = map_ptr[spec_idx01_];
        Half4Texel t11 = map_ptr[spec_idx11_];

        Vec3f rgb;
        rgb[0] = w00 * to_float(t00.c0) + w10 * to_float(t10.c0) + w01 * to_float(t01.c0) + w11 * to_float(t11.c0);
        rgb[1] = w00 * to_float(t00.c1) + w10 * to_float(t10.c1) + w01 * to_float(t01.c1) + w11 * to_float(t11.c1);
        rgb[2] = w00 * to_float(t00.c2) + w10 * to_float(t10.c2) + w01 * to_float(t01.c2) + w11 * to_float(t11.c2);
        return rgb;
    }

    // Caller guarantees density > 0 (the cheap density gate runs first in
    // integrate_cell). This isolates the expensive bilinear-lookup + sigmoid
    // path so it can be skipped on low-weight cells without re-checking
    // density.
    __device__ Vec3f compute_color(uint32_t point_idx, float t_0, float t_1, const Vec3f &current_point)
    {
        // Probe the diffuse octmap at the entry-face surface point (t_0)
        Vec3f hit = ray_.origin + t_0 * ray_.direction;
        Vec3f d = hit - current_point;
        float len = d.norm();
        if (len > 1e-8f)
            d /= len;
        else
            d = ray_.direction;

        const Half4Texel *diff_ptr = diffuse_ + point_idx * single_map_size_diff_;
        const Half4Texel *spec_ptr = specular_ + point_idx * single_map_size_spec_;

        Vec2f uv_diff = oct_encode<diff_map_res>(d);
        Vec3f diffuse_raw = bilinear_lookup(diff_ptr, uv_diff);
        Vec3f specular_raw = specular_bilinear_lookup(spec_ptr);

        Vec3f rgb_logit = diffuse_raw + specular_raw;
        Vec3f rgb;
        for (int c = 0; c < 3; c++)
            rgb[c] = 1.0f / (1.0f + expf(-rgb_logit[c]));
        return rgb;
    }
};

template <int diff_map_res, int spec_map_res> class RayIntegratorOctMapInferQ8
{
  public:
    __device__ RayIntegratorOctMapInferQ8(Ray ray,
                                          const uint8_t *__restrict__ diffuse,
                                          const uint8_t *__restrict__ specular,
                                          const float *__restrict__ density,
                                          float weight_threshold,
                                          float diff_scale,
                                          float diff_offset,
                                          float spec_scale,
                                          float spec_offset,
                                          float cell_skip_threshold)
        : ray_(ray), diffuse_(diffuse), specular_(specular), density_(density), weight_threshold_(weight_threshold),
          cell_skip_threshold_(cell_skip_threshold), diff_scale_(diff_scale), diff_offset_(diff_offset),
          spec_scale_(spec_scale), spec_offset_(spec_offset)
    {
        init_specular_bilinear(oct_encode<spec_map_res>(-ray.direction));
    }

    __device__ Vec3f get_accumulated_rgb() const { return accumulated_rgb_; }

    __device__ float get_transmittance() const { return transmittance_; }

    __device__ bool integrate_cell(uint32_t point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point,
                                   const Vec3f &next_point,
                                   float * /* unused */)
    {
        // Cheap density gate first.
        float s = density_[point_idx];
        if (s <= 1e-6f)
        {
            return transmittance_ > weight_threshold_;
        }

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1.0f - expf(-s * delta_t);
        float weight = transmittance_ * alpha;

        // Skip the q8 bilinear-lookup + dequant + sigmoid on cells with
        // negligible color contribution. Still advance transmittance.
        if (weight >= cell_skip_threshold_)
        {
            Vec3f rgb = compute_color(point_idx, t_0, t_1, current_point);
            accumulated_rgb_ += weight * rgb;
        }
        transmittance_ *= (1.0f - alpha);

        return transmittance_ > weight_threshold_;
    }

  private:
    Vec3f accumulated_rgb_ = Vec3f::Zero();
    float transmittance_ = 1.0f;
    Ray ray_;
    const uint8_t *__restrict__ diffuse_ = nullptr;
    const uint8_t *__restrict__ specular_ = nullptr;
    const float *__restrict__ density_ = nullptr;

    float weight_threshold_ = 1e-6f;
    float cell_skip_threshold_ = 0.0f;
    float diff_scale_, diff_offset_;
    float spec_scale_, spec_offset_;

    int spec_idx00_, spec_idx10_, spec_idx01_, spec_idx11_;
    float spec_fu_, spec_fv_;

    static constexpr int single_map_size_diff_ = 3 * diff_map_res * diff_map_res;
    static constexpr int single_map_size_spec_ = 3 * spec_map_res * spec_map_res;

    __device__ float dequant(uint8_t val, float scale, float offset) { return scale * (float)val + offset; }

    __device__ void init_specular_bilinear(Vec2f uv)
    {
        float u = uv[0];
        float v = uv[1];
        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);

        spec_idx00_ = oct_wrap_index<spec_map_res>(u0, v0);
        spec_idx10_ = oct_wrap_index<spec_map_res>(u0 + 1, v0);
        spec_idx01_ = oct_wrap_index<spec_map_res>(u0, v0 + 1);
        spec_idx11_ = oct_wrap_index<spec_map_res>(u0 + 1, v0 + 1);
        spec_fu_ = u - u0;
        spec_fv_ = v - v0;
    }

    __device__ Vec3f bilinear_lookup(const uint8_t *map_ptr, Vec2f uv, float scale, float offset)
    {
        float u = uv[0];
        float v = uv[1];

        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);
        int u1 = u0 + 1;
        int v1 = v0 + 1;
        float fu = u - u0;
        float fv = v - v0;

        int idx00 = oct_wrap_index<diff_map_res>(u0, v0);
        int idx10 = oct_wrap_index<diff_map_res>(u1, v0);
        int idx01 = oct_wrap_index<diff_map_res>(u0, v1);
        int idx11 = oct_wrap_index<diff_map_res>(u1, v1);

        float w00 = (1.0f - fu) * (1.0f - fv);
        float w10 = fu * (1.0f - fv);
        float w01 = (1.0f - fu) * fv;
        float w11 = fu * fv;

        Vec3f rgb = Vec3f::Zero();
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            float val = w00 * dequant(map_ptr[idx00 * 3 + c], scale, offset) +
                        w10 * dequant(map_ptr[idx10 * 3 + c], scale, offset) +
                        w01 * dequant(map_ptr[idx01 * 3 + c], scale, offset) +
                        w11 * dequant(map_ptr[idx11 * 3 + c], scale, offset);
            rgb[c] = val;
        }
        return rgb;
    }

    __device__ Vec3f specular_bilinear_lookup(const uint8_t *map_ptr, float scale, float offset)
    {
        float w00 = (1.0f - spec_fu_) * (1.0f - spec_fv_);
        float w10 = spec_fu_ * (1.0f - spec_fv_);
        float w01 = (1.0f - spec_fu_) * spec_fv_;
        float w11 = spec_fu_ * spec_fv_;

        Vec3f rgb = Vec3f::Zero();
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            float val = w00 * dequant(map_ptr[spec_idx00_ * 3 + c], scale, offset) +
                        w10 * dequant(map_ptr[spec_idx10_ * 3 + c], scale, offset) +
                        w01 * dequant(map_ptr[spec_idx01_ * 3 + c], scale, offset) +
                        w11 * dequant(map_ptr[spec_idx11_ * 3 + c], scale, offset);
            rgb[c] = val;
        }
        return rgb;
    }

    // Caller guarantees density > 0 (gate done in integrate_cell).
    __device__ Vec3f compute_color(uint32_t point_idx, float t_0, float t_1, const Vec3f &current_point)
    {
        // Probe the diffuse octmap at the entry-face surface point (t_0)
        Vec3f hit = ray_.origin + t_0 * ray_.direction;
        Vec3f d = hit - current_point;
        float len = d.norm();
        if (len > 1e-8f)
            d /= len;
        else
            d = ray_.direction;

        const uint8_t *diff_ptr = diffuse_ + point_idx * single_map_size_diff_;
        const uint8_t *spec_ptr = specular_ + point_idx * single_map_size_spec_;

        Vec2f uv_diff = oct_encode<diff_map_res>(d);
        Vec3f diffuse_raw = bilinear_lookup(diff_ptr, uv_diff, diff_scale_, diff_offset_);
        Vec3f specular_raw = specular_bilinear_lookup(spec_ptr, spec_scale_, spec_offset_);

        Vec3f rgb_logit = diffuse_raw + specular_raw;
        Vec3f rgb;
        for (int c = 0; c < 3; c++)
            rgb[c] = 1.0f / (1.0f + expf(-rgb_logit[c]));
        return rgb;
    }
};

template <typename attr_scalar, int diff_map_res, int spec_map_res> class RayIntegratorOctMapBackward
{
  public:
    __device__ RayIntegratorOctMapBackward(Ray ray,
                                           const attr_scalar *__restrict__ diffuse,
                                           const attr_scalar *__restrict__ specular,
                                           const float *__restrict__ density,
                                           float weight_threshold,
                                           Vec4f fwd_rgba,
                                           Vec4f fwd_rgba_grad,
                                           const float *ray_depth_grad,
                                           const float *quantile_thresholds,
                                           uint32_t num_depth_quantiles,
                                           float initial_depth_grad,
                                           float ray_error,
                                           float fwd_W_total,
                                           float fwd_S_total,
                                           float fwd_t_far,
                                           float fwd_distortion,
                                           float fwd_distortion_grad,
                                           Vec3f *points_grad,
                                           attr_scalar *diffuse_grad,
                                           attr_scalar *specular_grad,
                                           float *density_grad)
        : ray_(ray), diffuse_(diffuse), specular_(specular), density_(density), weight_threshold_(weight_threshold),
          quantile_thresholds_(quantile_thresholds), num_depth_quantiles_(num_depth_quantiles),
          ray_depth_grad_(ray_depth_grad), points_grad_(points_grad), diffuse_grad_(diffuse_grad),
          specular_grad_(specular_grad), density_grad_(density_grad), ray_error_(ray_error),
          W_total_(fwd_W_total), S_total_(fwd_S_total), t_far_(fwd_t_far),
          distortion_grad_(fwd_distortion_grad),
          // D_total = Σⱼ ∂L_outer/∂wⱼ · wⱼ = 2 * L_dist * distortion_grad (closed form derivation).
          D_total_(2.0f * fwd_distortion * fwd_distortion_grad)
    {
        fwd_rgb_ = fwd_rgba.template head<3>();
        fwd_alpha_ = fwd_rgba[3];
        fwd_rgb_grad_ = fwd_rgba_grad.template head<3>();
        fwd_alpha_grad_ = fwd_rgba_grad[3];

        current_depth_grad_ = initial_depth_grad;

        if (num_depth_quantiles_ > 0)
        {
            current_quantile_value_ = quantile_thresholds_[0];
        }

        init_specular_bilinear(oct_encode<spec_map_res>(-ray.direction));
    }

    __device__ bool integrate_cell(uint32_t point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point,
                                   const Vec3f &next_point,
                                   attr_scalar *point_error)
    {
        float s_primal = density_[point_idx];

        // Recompute forward values
        Vec3f rgb_primal = Vec3f::Zero();
        Vec2f uv_diff;
        bool has_color = false;

        if (s_primal > 1e-6f)
        {
            // Probe the diffuse octmap at the entry-face surface point (t_0)
            Vec3f hit = ray_.origin + t_0 * ray_.direction;
            Vec3f d = hit - current_point;
            float len = d.norm();
            if (len > 1e-8f)
                d /= len;
            else
                d = ray_.direction;

            const attr_scalar *diff_ptr = diffuse_ + point_idx * single_map_size_diff_;
            const attr_scalar *spec_ptr = specular_ + point_idx * single_map_size_spec_;

            uv_diff = oct_encode<diff_map_res>(d);
            Vec3f diffuse_raw = bilinear_lookup(diff_ptr, uv_diff);
            Vec3f specular_raw = specular_bilinear_lookup(spec_ptr);

            Vec3f rgb_logit = diffuse_raw + specular_raw;
            for (int c = 0; c < 3; c++)
                rgb_primal[c] = 1.0f / (1.0f + expf(-rgb_logit[c]));
            has_color = true;
        }

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);
        float weight = transmittance_ * alpha;

        float dalpha_ds_primal = delta_t * (1 - alpha);
        float dalpha_ddelta_t = 0.0f;
        if (delta_t > 0.0f)
        {
            dalpha_ddelta_t = s_primal * (1 - alpha);
        }

        accumulated_rgb_ += weight * rgb_primal;
        if (point_error)
        {
            atomicAdd(point_error + point_idx, from_float<attr_scalar>(weight * ray_error_));
        }

        Vec3f dL_drgb_primal = fwd_rgb_grad_ * weight;

        Vec3f rgb_rest = fwd_rgb_ - accumulated_rgb_;
        rgb_rest /= (transmittance_ * (1 - alpha + 1e-6f));

        float dL_dalpha = transmittance_ * (rgb_primal - rgb_rest).dot(fwd_rgb_grad_);
        dL_dalpha += (1 - fwd_alpha_) * fwd_alpha_grad_ / (1 - alpha + 1e-6f);

        float dL_ds_primal = dL_dalpha * dalpha_ds_primal;
        float dL_ddelta_t = dL_dalpha * dalpha_ddelta_t;

        float dL_dt0 = 0.0f;

        // Distortion loss gradient (mip-NeRF 360 eq. 15) in disparity-style s-space.
        // Distortion math uses s = 1 - 1/(1+t); chain rule into σ still uses raw δ_t
        // because ∂w_j/∂σ_i = -w_j · δ_t_raw,i (T factor along ray is in raw t).
        // Direct:   ∂L/∂σ_i (via w_i) = dL_dw_outer · T_i · δ_t_raw (1-α_i)
        // Indirect: ∂L/∂σ_i (via T_j for j>i) = -δ_t_raw · Σ_{j>i} dL_dw_outer_j · w_j
        //           Σ_{j>i} dL_dw_outer_j · w_j = D_total - D_running - dL_dw_outer_i · w_i.
        if (distortion_grad_ != 0.0f)
        {
            float s_0 = 1.0f - 1.0f / (1.0f + t_0);
            float s_1 = 1.0f - 1.0f / (1.0f + t_1);
            float mid_s = 0.5f * (s_0 + s_1);
            float delta_s = s_1 - s_0;
            float W_minus = W_run_;
            float S_minus = S_run_;
            float W_plus = W_total_ - W_minus - weight;
            float S_plus = S_total_ - S_minus - weight * mid_s;
            float dL_dw_raw = 2.0f * (mid_s * W_minus - S_minus) +
                              2.0f * (S_plus - mid_s * W_plus) +
                              (2.0f / 3.0f) * weight * delta_s;
            float dL_dw_outer = dL_dw_raw * distortion_grad_;
            float dL_dw_w = dL_dw_outer * weight;
            float D_iplus = D_total_ - D_running_ - dL_dw_w;
            dL_ds_primal += dL_dw_outer * transmittance_ * dalpha_ds_primal - delta_t * D_iplus;
            D_running_ += dL_dw_w;
            W_run_ += weight;
            S_run_ += weight * mid_s;
        }

        float next_transmittance = transmittance_ * (1 - alpha);
        while (current_quantile_idx_ < num_depth_quantiles_ && next_transmittance < current_quantile_value_)
        {
            float depth_grad_i = ray_depth_grad_[current_quantile_idx_] / s_primal;
            dL_dt0 += depth_grad_i;
            dL_ds_primal += -depth_grad_i * logf(transmittance_ / current_quantile_value_) / s_primal;

            current_depth_grad_ -= depth_grad_i;

            current_quantile_idx_++;
            if (current_quantile_idx_ < num_depth_quantiles_)
            {
                current_quantile_value_ = quantile_thresholds_[current_quantile_idx_];
            }
        }

        if (current_quantile_idx_ < num_depth_quantiles_)
        {
            dL_ds_primal += -delta_t * current_depth_grad_;
            dL_ddelta_t += -s_primal * current_depth_grad_;
        }

        dL_dt0 += -dL_ddelta_t;
        float dL_dt1 = dL_ddelta_t;

        // Position gradients through cell intersection geometry
        Vec3f dt0_dprev_point;
        if (prev_point_idx_ != UINT32_MAX)
        {
            dt0_dprev_point = cell_intersection_grad(prev_point_geom_, current_point, ray_);
        }
        else
        {
            dt0_dprev_point = Vec3f::Zero();
        }

        Vec3f dt1_dcurrent_point = cell_intersection_grad(current_point, next_point, ray_);
        Vec3f dt0_dcurrent_point = cell_intersection_grad(current_point, prev_point_geom_, ray_);
        Vec3f dt1_dnext_point = cell_intersection_grad(next_point, current_point, ray_);

        prev_point_grad_ += dL_dt0 * dt0_dprev_point;
        current_point_grad_ += dL_dt0 * dt0_dcurrent_point + dL_dt1 * dt1_dcurrent_point;
        next_point_grad_ += dL_dt1 * dt1_dnext_point;

        if (prev_point_idx_ != UINT32_MAX)
        {
            atomic_add_vec(points_grad_ + prev_point_idx_, prev_point_grad_);
        }
        prev_point_geom_ = current_point;
        prev_point_idx_ = point_idx;
        prev_point_grad_ = current_point_grad_;

        current_point_grad_ = next_point_grad_;
        next_point_grad_ = Vec3f::Zero();

        transmittance_ = next_transmittance;

        // Scatter texture gradients. Both maps flow through the same sigmoid
        // via the logit sum, so their chain-rule scalar is identical:
        //   dL/d_logit = dL/d_rgb * sigmoid * (1 - sigmoid)
        if (has_color)
        {
            Vec3f dL_dlogit;
            for (int c = 0; c < 3; c++)
                dL_dlogit[c] = dL_drgb_primal[c] * rgb_primal[c] * (1.0f - rgb_primal[c]);
            scatter_texel_grad(diffuse_grad_ + point_idx * single_map_size_diff_, uv_diff, dL_dlogit);
            scatter_specular_grad(specular_grad_ + point_idx * single_map_size_spec_, dL_dlogit);
        }

        // Density gradient (always fp32)
        atomicAdd(density_grad_ + point_idx, dL_ds_primal);

        if (transmittance_ < weight_threshold_)
        {
            atomic_add_vec(points_grad_ + prev_point_idx_, prev_point_grad_);
            return false;
        }
        else
        {
            return true;
        }
    }

  private:
    Vec3f accumulated_rgb_ = Vec3f::Zero();
    float transmittance_ = 1.0f;
    float current_depth_grad_ = 0.0f;
    uint32_t current_quantile_idx_ = 0;
    float current_quantile_value_ = 0.0f;

    Vec3f *points_grad_ = nullptr;
    attr_scalar *diffuse_grad_ = nullptr;
    attr_scalar *specular_grad_ = nullptr;
    float *density_grad_ = nullptr;

    // For position gradient chain (cell intersection geometry)
    Vec3f prev_point_geom_ = Vec3f::Zero();
    Vec3f prev_point_grad_ = Vec3f::Zero();
    uint32_t prev_point_idx_ = UINT32_MAX;
    Vec3f current_point_grad_ = Vec3f::Zero();
    Vec3f next_point_grad_ = Vec3f::Zero();

    Vec3f fwd_rgb_ = Vec3f::Zero();
    Vec3f fwd_rgb_grad_ = Vec3f::Zero();
    float fwd_alpha_ = 0.0f;
    float fwd_alpha_grad_ = 0.0f;
    const float *ray_depth_grad_ = nullptr;
    Ray ray_;
    const attr_scalar *__restrict__ diffuse_ = nullptr;
    const attr_scalar *__restrict__ specular_ = nullptr;
    const float *__restrict__ density_ = nullptr;
    float ray_error_ = 0.0f;

    const float *quantile_thresholds_ = nullptr;
    uint32_t num_depth_quantiles_ = 0;

    float weight_threshold_ = 1e-6f;

    // Distortion loss backward state (mip-NeRF 360 eq. 15).
    float W_run_ = 0.0f;
    float S_run_ = 0.0f;
    float W_total_ = 0.0f;
    float S_total_ = 0.0f;
    float t_far_ = 0.0f;
    float distortion_grad_ = 0.0f;
    float D_total_ = 0.0f;
    float D_running_ = 0.0f;

    int spec_idx00_, spec_idx10_, spec_idx01_, spec_idx11_;
    float spec_fu_, spec_fv_;

    static constexpr int single_map_size_diff_ = 3 * diff_map_res * diff_map_res;
    static constexpr int single_map_size_spec_ = 3 * spec_map_res * spec_map_res;

    __device__ void init_specular_bilinear(Vec2f uv)
    {
        float u = uv[0];
        float v = uv[1];
        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);

        spec_idx00_ = oct_wrap_index<spec_map_res>(u0, v0);
        spec_idx10_ = oct_wrap_index<spec_map_res>(u0 + 1, v0);
        spec_idx01_ = oct_wrap_index<spec_map_res>(u0, v0 + 1);
        spec_idx11_ = oct_wrap_index<spec_map_res>(u0 + 1, v0 + 1);
        spec_fu_ = u - u0;
        spec_fv_ = v - v0;
    }

    __device__ Vec3f bilinear_lookup(const attr_scalar *map_ptr, Vec2f uv)
    {
        float u = uv[0];
        float v = uv[1];

        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);
        int u1 = u0 + 1;
        int v1 = v0 + 1;
        float fu = u - u0;
        float fv = v - v0;

        int idx00 = oct_wrap_index<diff_map_res>(u0, v0);
        int idx10 = oct_wrap_index<diff_map_res>(u1, v0);
        int idx01 = oct_wrap_index<diff_map_res>(u0, v1);
        int idx11 = oct_wrap_index<diff_map_res>(u1, v1);

        float w00 = (1.0f - fu) * (1.0f - fv);
        float w10 = fu * (1.0f - fv);
        float w01 = (1.0f - fu) * fv;
        float w11 = fu * fv;

        Vec3f rgb = Vec3f::Zero();
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            float val = w00 * to_float(map_ptr[idx00 * 3 + c]) + w10 * to_float(map_ptr[idx10 * 3 + c]) +
                        w01 * to_float(map_ptr[idx01 * 3 + c]) + w11 * to_float(map_ptr[idx11 * 3 + c]);
            rgb[c] = val;
        }
        return rgb;
    }

    __device__ Vec3f specular_bilinear_lookup(const attr_scalar *map_ptr)
    {
        float w00 = (1.0f - spec_fu_) * (1.0f - spec_fv_);
        float w10 = spec_fu_ * (1.0f - spec_fv_);
        float w01 = (1.0f - spec_fu_) * spec_fv_;
        float w11 = spec_fu_ * spec_fv_;

        Vec3f rgb = Vec3f::Zero();
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            float val = w00 * to_float(map_ptr[spec_idx00_ * 3 + c]) + w10 * to_float(map_ptr[spec_idx10_ * 3 + c]) +
                        w01 * to_float(map_ptr[spec_idx01_ * 3 + c]) + w11 * to_float(map_ptr[spec_idx11_ * 3 + c]);
            rgb[c] = val;
        }
        return rgb;
    }

    __device__ void scatter_texel_grad(attr_scalar *map_grad_ptr, Vec2f uv, Vec3f dL_drgb)
    {
        float u = uv[0];
        float v = uv[1];

        int u0 = (int)floorf(u);
        int v0 = (int)floorf(v);
        int u1 = u0 + 1;
        int v1 = v0 + 1;
        float fu = u - u0;
        float fv = v - v0;

        int idx00 = oct_wrap_index<diff_map_res>(u0, v0);
        int idx10 = oct_wrap_index<diff_map_res>(u1, v0);
        int idx01 = oct_wrap_index<diff_map_res>(u0, v1);
        int idx11 = oct_wrap_index<diff_map_res>(u1, v1);

        float w00 = (1.0f - fu) * (1.0f - fv);
        float w10 = fu * (1.0f - fv);
        float w01 = (1.0f - fu) * fv;
        float w11 = fu * fv;

#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            atomicAdd(map_grad_ptr + idx00 * 3 + c, from_float<attr_scalar>(w00 * dL_drgb[c]));
            atomicAdd(map_grad_ptr + idx10 * 3 + c, from_float<attr_scalar>(w10 * dL_drgb[c]));
            atomicAdd(map_grad_ptr + idx01 * 3 + c, from_float<attr_scalar>(w01 * dL_drgb[c]));
            atomicAdd(map_grad_ptr + idx11 * 3 + c, from_float<attr_scalar>(w11 * dL_drgb[c]));
        }
    }

    __device__ void scatter_specular_grad(attr_scalar *map_grad_ptr, Vec3f dL_drgb)
    {
        float w00 = (1.0f - spec_fu_) * (1.0f - spec_fv_);
        float w10 = spec_fu_ * (1.0f - spec_fv_);
        float w01 = (1.0f - spec_fu_) * spec_fv_;
        float w11 = spec_fu_ * spec_fv_;

#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            atomicAdd(map_grad_ptr + spec_idx00_ * 3 + c, from_float<attr_scalar>(w00 * dL_drgb[c]));
            atomicAdd(map_grad_ptr + spec_idx10_ * 3 + c, from_float<attr_scalar>(w10 * dL_drgb[c]));
            atomicAdd(map_grad_ptr + spec_idx01_ * 3 + c, from_float<attr_scalar>(w01 * dL_drgb[c]));
            atomicAdd(map_grad_ptr + spec_idx11_ * 3 + c, from_float<attr_scalar>(w11 * dL_drgb[c]));
        }
    }
};

} // namespace vorotracing
