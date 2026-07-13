from typing import Callable

import torch


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


class ErrorBox:
    def __init__(self):
        self.ray_error = None
        self.point_error = None
        self.cell_distortion_gate = None


class TraceVoroTracing(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _points,
        _diffuse,
        _specular,
        _density,
        _point_adjacency,
        _point_adjacency_offsets,
        rays,
        start_point,
        depth_quantiles,
        weight_threshold: float,
        max_intersections: int,
        return_contribution: bool,
    ):
        ctx.rays = rays
        ctx.start_point = start_point
        ctx.depth_quantiles = depth_quantiles
        ctx.weight_threshold = weight_threshold
        ctx.max_intersections = max_intersections
        ctx.points = _points
        ctx.diffuse = _diffuse
        ctx.specular = _specular
        ctx.density = _density
        ctx.point_adjacency = _point_adjacency
        ctx.point_adjacency_offsets = _point_adjacency_offsets

        results = _make_lazy_cuda_func("trace_vorotracing_fwd")(
            _points,
            _diffuse,
            _specular,
            _density,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            depth_quantiles,
            weight_threshold,
            max_intersections,
            return_contribution,
        )
        (
            rgba,
            depth,
            depth_indices,
            contribution,
            num_intersections,
            distortion,
            W_total,
            S_total,
            t_far,
        ) = results

        ctx.rgba = rgba
        ctx.depth_indices = depth_indices
        ctx.distortion = distortion
        ctx.W_total = W_total
        ctx.S_total = S_total
        ctx.t_far = t_far

        errbox = ErrorBox()
        ctx.errbox = errbox

        return (rgba, depth, contribution, num_intersections, errbox, distortion)

    @staticmethod
    def backward(
        ctx,
        grad_rgba,
        grad_depth,
        grad_contribution,
        grad_num_intersections,
        errbox_grad,
        grad_distortion,
    ):
        del grad_num_intersections
        del errbox_grad

        rays = ctx.rays
        start_point = ctx.start_point
        rgba = ctx.rgba
        _points = ctx.points
        _diffuse = ctx.diffuse
        _specular = ctx.specular
        _density = ctx.density
        _point_adjacency = ctx.point_adjacency
        _point_adjacency_offsets = ctx.point_adjacency_offsets
        depth_quantiles = ctx.depth_quantiles

        contribution_alpha_grad = 0.0
        if grad_contribution is not None:
            contribution_alpha_grad = float(
                grad_contribution.detach().abs().max().item()
            )

        def _bwd(grad_rgba_, grad_depth_, ray_error_, contrib_, grad_distortion_):
            return _make_lazy_cuda_func("trace_vorotracing_bwd")(
                _points,
                _diffuse,
                _specular,
                _density,
                _point_adjacency,
                _point_adjacency_offsets,
                rays,
                start_point,
                rgba,
                grad_rgba_,
                depth_quantiles,
                ctx.depth_indices,
                grad_depth_,
                ray_error_,
                contrib_,
                ctx.distortion,
                grad_distortion_,
                ctx.W_total,
                ctx.S_total,
                ctx.t_far,
                ctx.weight_threshold,
                ctx.max_intersections,
            )

        cell_gate = getattr(ctx.errbox, "cell_distortion_gate", None)
        if cell_gate is not None and grad_distortion is not None:
            res_d = _bwd(
                torch.zeros_like(grad_rgba), None, None, 0.0, grad_distortion
            )
            density_grad_dist = res_d[3]
            res_m = _bwd(
                grad_rgba,
                grad_depth,
                ctx.errbox.ray_error,
                contribution_alpha_grad,
                torch.zeros_like(grad_distortion),
            )
            points_grad, diffuse_grad, specular_grad, density_grad, ray_grad, point_error = (
                res_m
            )
            torch.nan_to_num(
                density_grad_dist,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
                out=density_grad_dist,
            )
            density_grad = (
                density_grad + cell_gate.reshape_as(density_grad) * density_grad_dist
            )
        else:
            results = _bwd(
                grad_rgba,
                grad_depth,
                ctx.errbox.ray_error,
                contribution_alpha_grad,
                grad_distortion,
            )
            (
                points_grad,
                diffuse_grad,
                specular_grad,
                density_grad,
                ray_grad,
                point_error,
            ) = results
        ctx.errbox.point_error = point_error

        torch.nan_to_num(points_grad, nan=0.0, posinf=0.0, neginf=0.0, out=points_grad)
        torch.nan_to_num(
            diffuse_grad, nan=0.0, posinf=0.0, neginf=0.0, out=diffuse_grad
        )
        torch.nan_to_num(
            specular_grad, nan=0.0, posinf=0.0, neginf=0.0, out=specular_grad
        )
        torch.nan_to_num(
            density_grad, nan=0.0, posinf=0.0, neginf=0.0, out=density_grad
        )

        del (
            ctx.rays,
            ctx.start_point,
            ctx.rgba,
            ctx.points,
            ctx.diffuse,
            ctx.specular,
            ctx.density,
            ctx.point_adjacency,
            ctx.point_adjacency_offsets,
            ctx.depth_quantiles,
            ctx.weight_threshold,
            ctx.max_intersections,
            ctx.W_total,
            ctx.S_total,
            ctx.t_far,
        )
        del ctx.distortion
        return (
            points_grad,
            diffuse_grad,
            specular_grad,
            density_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def prefetch_octmap_adj(points, adjacency, offsets):
    return _make_lazy_cuda_func("trace_vorotracing_prefetch_adj")(
        points, adjacency, offsets
    )


def trace_octmap_infer(
    points,
    diffuse_fp16,
    specular_fp16,
    density,
    adjacency,
    offsets,
    rays,
    start_point,
    adjacent_diff,
    ray_perm,
    weight_threshold=0.001,
    max_intersections=1024,
    cell_skip_threshold=0.0,
):
    return _make_lazy_cuda_func("trace_vorotracing_infer")(
        points,
        diffuse_fp16,
        specular_fp16,
        density,
        adjacency,
        offsets,
        rays,
        start_point,
        adjacent_diff,
        ray_perm,
        weight_threshold,
        max_intersections,
        cell_skip_threshold,
    )


def trace_octmap_infer_q8(
    points,
    diffuse_q8,
    specular_q8,
    density,
    adjacency,
    offsets,
    rays,
    start_point,
    adjacent_diff,
    diff_scale,
    diff_offset,
    spec_scale,
    spec_offset,
    weight_threshold=0.001,
    max_intersections=1024,
    cell_skip_threshold=0.0,
):
    return _make_lazy_cuda_func("trace_vorotracing_infer_q8")(
        points,
        diffuse_q8,
        specular_q8,
        density,
        adjacency,
        offsets,
        rays,
        start_point,
        adjacent_diff,
        diff_scale,
        diff_offset,
        spec_scale,
        spec_offset,
        weight_threshold,
        max_intersections,
        cell_skip_threshold,
    )
