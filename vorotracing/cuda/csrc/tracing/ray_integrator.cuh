#pragma once

#include <cuda_fp16.h>

#include "../utils/geometry.h"
#include "../utils/typing.cuh"
#include <assert.h>
#include <cuda_runtime.h>

namespace vorotracing
{

__constant__ float C0 = 0.28209479177387814f;
__constant__ float C1 = 0.4886025119029199f;
__constant__ float C2[5] = {1.0925484305920792f,
                            -1.0925484305920792f,
                            0.31539156525252005f,
                            -1.0925484305920792f,
                            0.5462742152960396f};
__constant__ float C3[7] = {-0.5900435899266435f,
                            2.890611442640554f,
                            -0.4570457994644658f,
                            0.3731763325901154f,
                            -0.4570457994644658f,
                            1.445305721320277f,
                            -0.5900435899266435f};
__constant__ float C4[9] = {2.5033429417967046f,
                            -1.7701307697799304f,
                            0.9461746957575601f,
                            -0.6690465435572892f,
                            0.10578554691520431f,
                            -0.6690465435572892f,
                            0.47308734787878004f,
                            -1.7701307697799304f,
                            0.6258357354491761f};

constexpr int sh_dimension(int degree) { return (degree + 1) * (degree + 1); }

__forceinline__ __device__ Vec3f cell_intersection_grad(const Vec3f &primal_point,
                                                        const Vec3f &opposite_point,
                                                        const Ray &ray)
{
    Vec3f face_origin = (primal_point + opposite_point) / 2.0f;
    Vec3f face_normal = (opposite_point - primal_point);

    float num = (face_origin - ray.origin).dot(face_normal);
    float dp = face_normal.dot(ray.direction);

    Vec3f grad = num * ray.direction + dp * (ray.origin - primal_point);
    grad /= dp * dp;

    return grad;
}

/// @brief Base class for ray integrators
/// @tparam attr_scalar The type of the attributes
/// This class is used to be able to pass different ray integrators to the generic tracing function.
template <typename attr_scalar> class RayIntegrator
{
  public:
    __device__ virtual bool integrate_cell(uint32_t point_idx,
                                           float t_0,
                                           float t_1,
                                           const Vec3f &current_point,
                                           const Vec3f &next_point,
                                           const attr_scalar *attributes,
                                           attr_scalar *debug_output)
    {
        return false;
    }
};

/// @brief Ray integrator for SH-based rendering (Forward pass)
/// @tparam attr_scalar The type of the attributes
/// @tparam sh_degree The degree of the SH basis functions
template <typename attr_scalar, int sh_degree> class RayIntegratorSH : public RayIntegrator<attr_scalar>
{
  public:
    __device__ RayIntegratorSH(Vec3f view_direction,
                               float weight_threshold,
                               const float *quantile_thresholds,
                               uint32_t num_depth_quantiles,
                               float *quantile_depths,
                               uint32_t *quantile_point_indices)
        : weight_threshold_(weight_threshold), quantile_thresholds_(quantile_thresholds),
          num_depth_quantiles_(num_depth_quantiles), quantile_depths_(quantile_depths),
          quantile_point_indices_(quantile_point_indices)
    {
        // We precompute the SH coefficients for the view direction avoiding recomputing them for each cell.
        sh_coeffs_ = sh_coefficients(view_direction);

        // If we want to compute depth quantiles, all pointers must be valid
        assert(num_depth_quantiles_ == 0 ||
               (quantile_thresholds_ != nullptr && quantile_depths_ != nullptr && quantile_point_indices_ != nullptr));

        if (num_depth_quantiles_ > 0)
        {
            current_quantile_value_ = quantile_thresholds_[0];
        }
    }

    __device__ Vec3f get_accumulated_rgb() const { return accumulated_rgb_; }

    __device__ float get_transmittance() const { return transmittance_; }

    __device__ uint32_t get_num_filled_quantiles() const { return current_quantile_idx_; }

    __device__ bool integrate_cell(uint32_t point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point, // Unused in forward pass
                                   const Vec3f &next_point,    // Unused in forward pass
                                   const attr_scalar *attributes,
                                   attr_scalar *point_contribution)
    {
        Vec3f rgb_primal;
        float s_primal;

        load_attributes(point_idx, rgb_primal, s_primal, attributes);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);
        float weight = transmittance_ * alpha;

        if (point_contribution)
        {
            atomicAdd(point_contribution + point_idx, from_float<attr_scalar>(weight));
        }
        accumulated_rgb_ += weight * rgb_primal;

        float next_transmittance = transmittance_ * (1 - alpha);
        while (current_quantile_idx_ < num_depth_quantiles_ && next_transmittance < current_quantile_value_)
        {
            quantile_depths_[current_quantile_idx_] = t_0 + logf(transmittance_ / current_quantile_value_) / s_primal;
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
    Vecf<sh_dimension(sh_degree)> sh_coeffs_;

    // Quantiles of the transmittance and the corresponding depths
    const float *quantile_thresholds_ = nullptr; // [Input] Defined transmittance values for each quantile
    uint32_t num_depth_quantiles_ = 0;           // [Input] Number quantiles
    float *quantile_depths_ = nullptr;           // [Output] Depths measured at each quantile
    uint32_t *quantile_point_indices_ = nullptr; // [Output] Point indices at each quantile

    uint32_t current_quantile_idx_ = 0;
    float current_quantile_value_ = 0.0f;

    float weight_threshold_ = 1e-6f;
    const int sh_memory_size_ = 3 * (1 + sh_degree) * (1 + sh_degree) + 1;

    __device__ void load_attributes(uint32_t point_idx, Vec3f &rgb, float &s, const attr_scalar *attributes)
    {
        const attr_scalar *attr_ptr = attributes + point_idx * sh_memory_size_;
        s = to_float(attr_ptr[sh_memory_size_ - 1]);
        if (s > 1e-6f)
        {
            // TODO: implement this function
            rgb = load_sh_as_rgb(attr_ptr);
        }
        else
        {
            rgb = Vec3f::Zero();
        }
    }

    __device__ Vec3f load_sh_as_rgb(const attr_scalar *sh_rgb_vals)
    {
        Vec3f rgb = Vec3f(0.5f, 0.5f, 0.5f);

#pragma unroll
        for (uint32_t i = 0; i < 3 * sh_dimension(sh_degree); ++i)
        {
            rgb[i % 3] += sh_coeffs_[i / 3] * to_float(sh_rgb_vals[i]);
        }

        return rgb.cwiseMax(0.0f);
    }

    __device__ Vecf<sh_dimension(sh_degree)> sh_coefficients(const Vec3f &dir)
    {
        float x = dir[0];
        float y = dir[1];
        float z = dir[2];

        Vecf<sh_dimension(sh_degree)> sh = Vecf<sh_dimension(sh_degree)>::Zero();

        sh[0] = C0;

        if (sh_degree > 0)
        {
            sh[1] = -C1 * y;
            sh[2] = C1 * z;
            sh[3] = -C1 * x;
        }
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        if (sh_degree > 1)
        {

            sh[4] = C2[0] * xy;
            sh[5] = C2[1] * yz;
            sh[6] = C2[2] * (2.0f * zz - xx - yy);
            sh[7] = C2[3] * xz;
            sh[8] = C2[4] * (xx - yy);
        }
        if (sh_degree > 2)
        {
            sh[9] = C3[0] * y * (3.0f * xx - yy);
            sh[10] = C3[1] * xy * z;
            sh[11] = C3[2] * y * (4.0f * zz - xx - yy);
            sh[12] = C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
            sh[13] = C3[4] * x * (4.0f * zz - xx - yy);
            sh[14] = C3[5] * z * (xx - yy);
            sh[15] = C3[6] * x * (xx - 3.0f * yy);
        }

        return sh;
    }
};

/// @brief Ray integrator for SH-based rendering (Backward pass)
/// @tparam attr_scalar The type of the attributes
/// @tparam sh_degree The degree of the SH basis functions
template <typename attr_scalar, int sh_degree> class RayIntegratorSHBackward : public RayIntegrator<attr_scalar>
{
  public:
    __device__ RayIntegratorSHBackward(Ray ray,
                                       float weight_threshold,
                                       Vec4f fwd_rgba,
                                       Vec4f fwd_rgba_grad,
                                       const float *ray_depth_grad,
                                       const float *quantile_thresholds,
                                       uint32_t num_depth_quantiles,
                                       float initial_depth_grad,
                                       float ray_error,             // Debug input
                                       Vec3f *points_grad,          // output
                                       attr_scalar *attribute_grad) // output
        : weight_threshold_(weight_threshold), quantile_thresholds_(quantile_thresholds),
          num_depth_quantiles_(num_depth_quantiles), ray_depth_grad_(ray_depth_grad), points_grad_(points_grad),
          attribute_grad_(attribute_grad), ray_error_(ray_error), ray_(ray)
    {
        fwd_rgb_ = fwd_rgba.template head<3>();
        fwd_alpha_ = fwd_rgba[3];
        fwd_rgb_grad_ = fwd_rgba_grad.template head<3>();
        fwd_alpha_grad_ = fwd_rgba_grad[3];

        current_depth_grad_ = initial_depth_grad; // set initial depth gradient

        sh_coeffs_ = sh_coefficients(ray.direction);
        if (num_depth_quantiles_ > 0)
        {
            current_quantile_value_ = quantile_thresholds_[0];
        }
    }

    __device__ bool integrate_cell(uint32_t point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point,
                                   const Vec3f &next_point,
                                   const attr_scalar *attributes,
                                   attr_scalar *point_error)
    {
        Vec3f rgb_primal;
        float s_primal; // Cell density

        load_attributes(point_idx, rgb_primal, s_primal, attributes);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);      // Cell thickness
        float alpha = 1 - expf(-s_primal * delta_t); // Cell opacity
        float weight = transmittance_ * alpha;       // Cell contribution

        float dalpha_ds_primal = delta_t * (1 - alpha); // Derivative of cell opacity with respect to cell density
        float dalpha_ddelta_t = 0.0f;                   // Derivative of cell opacity with respect to cell thickness
        if (delta_t > 0.0f)
        {
            dalpha_ddelta_t = s_primal * (1 - alpha);
        }

        accumulated_rgb_ += weight * rgb_primal;
        if (point_error)
        {
            atomicAdd(point_error + point_idx, from_float<attr_scalar>(weight * ray_error_));
        }

        Vec3f dL_drgb_primal = fwd_rgb_grad_ * weight; // dL_drgb_primal = dL_drgb * weight

        // dL_dalpha = dL_drgb * T * (rgb_primal - rgb_rest / (T*(1-alpha) ))
        Vec3f rgb_rest = fwd_rgb_ - accumulated_rgb_;
        rgb_rest /= (transmittance_ * (1 - alpha + 1e-6f));

        float dL_dalpha = transmittance_ * (rgb_primal - rgb_rest).dot(fwd_rgb_grad_);
        dL_dalpha += (1 - fwd_alpha_) * fwd_alpha_grad_ / (1 - alpha + 1e-6f);

        float dL_ds_primal = dL_dalpha * dalpha_ds_primal;
        float dL_ddelta_t = dL_dalpha * dalpha_ddelta_t;

        float dL_dt0 = 0.0f;

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

        Vec3f dt0_dprev_point;
        if (prev_point_idx_ != UINT32_MAX)
        {
            dt0_dprev_point = cell_intersection_grad(prev_point_, current_point, ray_);
        }
        else
        {
            dt0_dprev_point = Vec3f::Zero();
        }

        Vec3f dt1_dcurrent_point = cell_intersection_grad(current_point, next_point, ray_);
        Vec3f dt0_dcurrent_point = cell_intersection_grad(current_point, prev_point_, ray_);

        Vec3f dt1_dnext_point = cell_intersection_grad(next_point, current_point, ray_);

        prev_point_grad_ += dL_dt0 * dt0_dprev_point;
        current_point_grad_ += dL_dt0 * dt0_dcurrent_point + dL_dt1 * dt1_dcurrent_point;
        next_point_grad_ += dL_dt1 * dt1_dnext_point;

        if (prev_point_idx_ != UINT32_MAX)
        {
            atomic_add_vec(points_grad_ + prev_point_idx_, prev_point_grad_);
        }
        prev_point_ = current_point;
        prev_point_idx_ = point_idx;
        prev_point_grad_ = current_point_grad_;

        current_point_grad_ = next_point_grad_;
        next_point_grad_ = Vec3f::Zero();

        transmittance_ = next_transmittance;

        for (uint32_t i = 0; i < 3; ++i)
        {
            if (rgb_primal[i] == 0.0f)
            {
                dL_drgb_primal[i] = 0.0f;
            }
        }

        write_rgb_grad_to_sh(dL_drgb_primal,
                             attribute_grad_ + point_idx * sh_memory_size_); // write the gradient to the SH attributes

        atomicAdd(attribute_grad_ + point_idx * sh_memory_size_ + (sh_memory_size_ - 1),
                  from_float<attr_scalar>(dL_ds_primal)); // write the gradient to the density attribute

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
    // Accumulative/runtime values
    Vec3f accumulated_rgb_ = Vec3f::Zero();
    float transmittance_ = 1.0f;
    float current_depth_grad_ = 0.0f;
    uint32_t current_quantile_idx_ = 0;
    float current_quantile_value_ = 0.0f;

    // Gradients
    Vec3f *points_grad_ = nullptr;
    attr_scalar *attribute_grad_ = nullptr;

    Vec3f prev_point_ = Vec3f::Zero();
    Vec3f prev_point_grad_ = Vec3f::Zero();
    uint32_t prev_point_idx_ = UINT32_MAX;
    Vec3f current_point_grad_ = Vec3f::Zero();
    Vec3f next_point_grad_ = Vec3f::Zero();

    // external set values
    Vecf<sh_dimension(sh_degree)> sh_coeffs_;
    Vec3f fwd_rgb_ = Vec3f::Zero();
    Vec3f fwd_rgb_grad_ = Vec3f::Zero(); // dL_drgb
    float fwd_alpha_ = 0.0f;
    float fwd_alpha_grad_ = 0.0f;
    const float *ray_depth_grad_ = nullptr;
    const Ray &ray_;
    float ray_error_ = 0.0f;

    // Quantiles of the transmittance and the corresponding depths
    const float *quantile_thresholds_ = nullptr; // [Input] Defined transmittance values for each quantile
    uint32_t num_depth_quantiles_ = 0;           // [Input] Number quantiles

    float weight_threshold_ = 1e-6f;
    const int sh_memory_size_ = 3 * (1 + sh_degree) * (1 + sh_degree) + 1;

    __device__ void write_rgb_grad_to_sh(Vec3f grad_rgb, attr_scalar *sh_rgb_grad)
    {
        for (uint32_t i = 0; i < 3 * sh_dimension(sh_degree); ++i)
        {
            atomicAdd(sh_rgb_grad + i, from_float<attr_scalar>(sh_coeffs_[i / 3] * grad_rgb[i % 3]));
        }
    }

    __device__ void load_attributes(uint32_t point_idx, Vec3f &rgb, float &s, const attr_scalar *attributes)
    {
        const attr_scalar *attr_ptr = attributes + point_idx * sh_memory_size_;
        s = to_float(attr_ptr[sh_memory_size_ - 1]);
        if (s > 1e-6f)
        {
            // TODO: implement this function
            rgb = load_sh_as_rgb(attr_ptr);
        }
        else
        {
            rgb = Vec3f::Zero();
        }
    }

    __device__ Vec3f load_sh_as_rgb(const attr_scalar *sh_rgb_vals)
    {
        Vec3f rgb = Vec3f(0.5f, 0.5f, 0.5f);

#pragma unroll
        for (uint32_t i = 0; i < 3 * sh_dimension(sh_degree); ++i)
        {
            rgb[i % 3] += sh_coeffs_[i / 3] * to_float(sh_rgb_vals[i]);
        }

        return rgb.cwiseMax(0.0f);
    }

    __device__ Vecf<sh_dimension(sh_degree)> sh_coefficients(const Vec3f &dir)
    {
        float x = dir[0];
        float y = dir[1];
        float z = dir[2];

        Vecf<sh_dimension(sh_degree)> sh = Vecf<sh_dimension(sh_degree)>::Zero();

        sh[0] = C0;

        if (sh_degree > 0)
        {
            sh[1] = -C1 * y;
            sh[2] = C1 * z;
            sh[3] = -C1 * x;
        }
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        if (sh_degree > 1)
        {

            sh[4] = C2[0] * xy;
            sh[5] = C2[1] * yz;
            sh[6] = C2[2] * (2.0f * zz - xx - yy);
            sh[7] = C2[3] * xz;
            sh[8] = C2[4] * (xx - yy);
        }
        if (sh_degree > 2)
        {
            sh[9] = C3[0] * y * (3.0f * xx - yy);
            sh[10] = C3[1] * xy * z;
            sh[11] = C3[2] * y * (4.0f * zz - xx - yy);
            sh[12] = C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
            sh[13] = C3[4] * x * (4.0f * zz - xx - yy);
            sh[14] = C3[5] * z * (xx - yy);
            sh[15] = C3[6] * x * (xx - 3.0f * yy);
        }

        return sh;
    }
};

} // namespace vorotracing