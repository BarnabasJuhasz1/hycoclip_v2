#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from loguru import logger

_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
_cast_dtype = torch.float32
_enable_autocast = torch.cuda.is_available()

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def time_component(x: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    Compute the time component of a point on the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the time component of the input points.
    """
    return torch.sqrt(1.0 / curv + torch.sum(x.square(), dim=-1))

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """
    inv_curv = 1.0 / curv
    x_time = torch.sqrt(inv_curv + torch.sum(x.square(), dim=-1, keepdim=True))
    y_time = torch.sqrt(inv_curv + torch.sum(y.square(), dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / torch.sqrt(curv)

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """

    rc_xnorm = curv.sqrt() * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x.square(), dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv.sqrt() * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-6
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv.sqrt() + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """
    inv_curv = 1.0 / curv
    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_norm_square = torch.sum(x.square(), dim=-1)
    x_time = torch.sqrt(inv_curv + x_norm_square)
    y_time = torch.sqrt(inv_curv + torch.sum(y.square(), dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl.square() - 1, min=eps))

    acos_input = acos_numer / (x_norm_square.sqrt() * acos_denom + eps)
    acos_input = torch.clamp(acos_input, min=-1 + eps, max=1 - eps)
    _angle = torch.atan2(torch.sqrt(1 - acos_input * acos_input), acos_input)

    return _angle

##### ADDITIONAL HYPERBOLIC FUNCTIONS #####

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def pairwise_sim(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, dist_based=False):
    """
    Compute pairwise Lorentzian similarity between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian similarity
        between input vectors.
    """
    if dist_based:
        xyd = pairwise_dist(x, y, curv) # xyd in [0, inf)
        siml = 1./ (1. + xyd) # xyd in [0, 1]
    else:
        xyl = pairwise_inner(x, y, curv)
        siml = - 1. / (curv * xyl) # xyl in (inf, -1]
    return siml 

def elementwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    Compute the element-wise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B, D)` giving the space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B, D)` giving the space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B,)` giving the element-wise Lorentzian inner product
        between input vectors.
    """
    inv_curv = 1.0 / curv
    x_time = torch.sqrt(inv_curv + torch.sum(x.square(), dim=-1))
    y_time = torch.sqrt(inv_curv + torch.sum(y.square(), dim=-1))
    xyl = torch.sum(x * y, dim=-1) - x_time * y_time
    return xyl

def elementwise_dist(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Compute the element-wise geodesic distance between corresponding points
    in two batches of points on the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving the space components of a batch
            of points on the hyperboloid.
        y: Tensor of shape `(B, D)` giving the space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B,)` giving the element-wise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * elementwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv.sqrt()

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def log_map(s: Tensor, x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.
    This implementation is based on Eq.(20) of http://arxiv.org/abs/2303.15919.

    Args:
        s: Tensor of shape `(B, D)` giving space components of the point
            on the hyperboloid where the tangent space is defined.
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the point s.
    """
    if s.ndim == 1:
        s = s.unsqueeze(0)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    beta = -curv * elementwise_inner(s, x, curv).permute(1,0)
    cosh_input = torch.clamp(beta, min=eps, max=math.acosh(2**15))
    den = torch.clamp(torch.sqrt(beta.square() - 1), min=eps)

    _output = torch.acosh(cosh_input) / den * (x - beta * s)
    return _output

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def cosine_similarity(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6
) -> Tensor:
    """
    Compute the cosine similarity between two vectors on the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the cosine similarity between input vectors.
    """
    x_norm = x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)
    y_norm = y / torch.clamp(torch.norm(y, dim=-1, keepdim=True), min=eps)
    sim = x_norm @ y_norm.transpose(-1, -2)
    return sim

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def pairwise_oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    inv_curv = 1.0 / curv
    x_norm_square = torch.sum(x.square(), dim=-1) # (256,)
    x_time = torch.sqrt(inv_curv + x_norm_square)[:,None] # (256,1)
    y_time = torch.sqrt(inv_curv + torch.sum(y.square(), dim=-1))[None] # (1,512)

    # Calculate lorentzian inner product multiplied with curvature.
    c_xyl = curv * pairwise_inner(x, y, curv) # (256, 512)
    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time # (1,512) + (256,512)*(256,1) -> (256,512)
    acos_denom = torch.sqrt(torch.clamp(c_xyl.square() - 1, min=eps))

    acos_input = acos_numer / (x_norm_square[...,None] * acos_denom + eps)
    acos_input = torch.clamp(acos_input, min=-1 + eps, max=1 - eps)
    _angle = torch.atan2(torch.sqrt(1 - acos_input * acos_input), acos_input)

    return _angle

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def elementwise_distance_cone_border(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, margin: float = 0.1, eps: float = 1e-6
) -> Tensor:
    """
    Compute the distance of a point from the border of the entailment cone.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.
        margin: Margin to define the border of the entailment cone.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B, )` giving the distance from the cone border.
    """
    ext_angle = oxy_angle(x, y, curv, eps)
    with torch.no_grad():
        # avoid the following optimization shortcuts:
        # - minimize the hypotenuse to reduce the magnitude of the distance
        # - move the aperture border closer to the point instead of the point to the border (squash the cone apex's norm)
        hypotenuse = torch.sinh(elementwise_dist(x, y, curv, eps))
        half_aperture_angle = half_aperture(x, curv, margin, eps)
        
    margin = half_aperture_angle - margin if margin > 0 else 0
    residual_angle = torch.clamp(ext_angle, min=margin)
    d_border = torch.asinh(torch.sin(residual_angle) * hypotenuse)

    return d_border

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def pairwise_distance_cone_border(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, margin: float = 0.1, eps: float = 1e-6
) -> Tensor:
    """
    Compute the pairwise distance of points from the border of the entailment cone.

    Args:
        x: Tensor of shape `(B1, D)` giving space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving space components of another
            batch of vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        margin: Margin to define the border of the entailment cone.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distances from the cone border.
    """
    ext_angle = pairwise_oxy_angle(x, y, curv, eps)
    with torch.no_grad():
        # avoid the following optimization shortcuts:
        # - minimize the hypotenuse to reduce the magnitude of the distance
        # - move the aperture border closer to the point instead of the point to the border (squash the cone apex's norm)
        hypotenuse = torch.sinh(pairwise_dist(x, y, curv, eps))
        half_aperture_angle = half_aperture(x, curv, margin, eps)
        
    margin = half_aperture_angle - margin if margin > 0 else 0
    residual_angle = torch.clamp(ext_angle, min=margin)
    d_border = torch.asinh(torch.sin(residual_angle) * hypotenuse)

    return d_border

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def elementwise_chord_length(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6
) -> Tensor:
    """
    Compute the chord length between two points on the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B, )` giving the chord length between input points.
    """
    inner_xy = elementwise_inner(x, y, curv)
    first_term = torch.clamp(-curv * inner_xy, min=1 + eps)
    second_term = torch.sinh(torch.acosh(first_term))
    second_term = second_term.square() * torch.cos(oxy_angle(x, y, curv, eps))

    chord = torch.acosh(torch.clamp(first_term.square() - second_term, min=1 + eps)) / curv.sqrt()
    return chord

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def safe_elementwise_chord_length(x, y, curv=1.0, eps=1e-6):

    inner_xy = elementwise_inner(x, y, curv)
    first_term = torch.clamp(-curv * inner_xy, min=1.0 + eps, max=1e6)

    # sinh^2(z) = cosh^2(z) - 1
    first_term_square = first_term * first_term
    sinh_acosh = torch.clamp(first_term_square - 1.0, min=0.0)
    cos_theta = torch.cos(oxy_angle(x, y, curv, eps)).clamp(-1.0 + eps, 1.0 - eps)

    second_term = sinh_acosh * cos_theta
    z = torch.clamp(first_term_square - second_term, min=1.0 + eps)

    # acosh(z) = log(z + sqrt(z^2 - 1))
    chord = torch.log(z + torch.sqrt(z * z - 1.0 + eps)) / torch.sqrt(curv)
    return chord

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def safe_elementwise_chord_length_and_dist(x, y, curv=1.0, eps=1e-6):

    inner_xy = elementwise_inner(x, y, curv)
    first_term = torch.clamp(-curv * inner_xy, min=1.0 + eps, max=1e6)

    # sinh^2(z) = cosh^2(z) - 1
    first_term_square = first_term * first_term
    sinh_acosh = torch.clamp(first_term_square - 1.0, min=0.0)
    cos_theta = torch.cos(oxy_angle(x, y, curv, eps)).clamp(-1.0 + eps, 1.0 - eps)

    second_term = sinh_acosh * cos_theta
    z = torch.clamp(first_term_square - second_term, min=1.0 + eps)

    # acosh(z) = log(z + sqrt(z^2 - 1))
    chord = torch.log(z + torch.sqrt(z * z - 1.0 + eps)) / torch.sqrt(curv)
    d = torch.log(first_term + torch.sqrt(first_term_square - 1.0 + eps)) / torch.sqrt(curv)
    return 0.5 * chord + 0.5 * d

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def pairwise_chord_length_not_optimized(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6
) -> Tensor:
    """
    Compute the chord length between two points on the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B, )` giving the chord length between input points.
    """
    sqrt_curv = curv.sqrt()
    r = sqrt_curv * pairwise_dist(x, y, curv, eps) # (256,512)
    ext_angle_xy = pairwise_oxy_angle(x, y, curv, eps) # (256,512)

    chord = torch.cosh(r).square() - torch.sinh(r).square() * torch.cos(ext_angle_xy)
    chord = torch.acosh(torch.clamp(chord, min=1.0 + eps)) / sqrt_curv

    return chord

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def elementwise_chord_length_not_optimized(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6
) -> Tensor:
    """
    Compute the chord length between two points on the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B, )` giving the chord length between input points.
    """
    sqrt_curv = curv.sqrt()
    r = sqrt_curv * elementwise_dist(x, y, curv, eps) # (256,512)
    ext_angle_xy = oxy_angle(x, y, curv, eps) # (256,512)

    chord = torch.cosh(r).square() - torch.sinh(r).square() * torch.cos(ext_angle_xy)
    chord = torch.acosh(chord) / sqrt_curv

    return chord

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def pairwise_chord_length(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6
) -> Tensor:
    """
    Compute the chord length between two points on the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B, )` giving the chord length between input points.
    """
    inner_xy = pairwise_inner(x, y, curv)
    first_term = torch.clamp(-curv * inner_xy, min=1.0 + eps, max=1e6)
    second_term = torch.sinh(torch.acosh(first_term))
    second_term = second_term.square() * torch.cos(pairwise_oxy_angle(x, y, curv, eps))

    chord = torch.acosh(torch.clamp(first_term.square() - second_term, min=1 + eps)) / curv.sqrt()
    return chord

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def safe_pairwise_chord_length(x, y, curv=1.0, eps=1e-6):

    inner_xy = pairwise_inner(x, y, curv)
    first_term = torch.clamp(-curv * inner_xy, min=1.0 + eps, max=1e6)

    # sinh^2(z) = cosh^2(z) - 1
    first_term_square = first_term * first_term
    sinh_acosh = torch.clamp(first_term_square - 1.0, min=0.0)
    cos_theta = torch.cos(pairwise_oxy_angle(x, y, curv, eps)).clamp(-1.0 + eps, 1.0 - eps)

    second_term = sinh_acosh * cos_theta
    z = torch.clamp(first_term_square - second_term, min=1.0 + eps)

    # acosh(z) = log(z + sqrt(z^2 - 1))
    chord = torch.log(z + torch.sqrt(z * z - 1.0)) / torch.sqrt(curv)
    return chord

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def safe_pairwise_chord_length_and_dist(x, y, curv=1.0, eps=1e-6):

    inner_xy = pairwise_inner(x, y, curv)
    first_term = torch.clamp(-curv * inner_xy, min=1.0 + eps, max=1e6)
    
    # sinh^2(z) = cosh^2(z) - 1
    first_term_square = first_term * first_term
    sinh_acosh = torch.clamp(first_term_square - 1.0, min=0.0)
    cos_theta = torch.cos(pairwise_oxy_angle(x, y, curv, eps)).clamp(-1.0 + eps, 1.0 - eps)

    second_term = sinh_acosh * cos_theta
    z = torch.clamp(first_term_square - second_term, min=1.0 + eps)

    # acosh(z) = log(z + sqrt(z^2 - 1))
    chord = torch.log(z + torch.sqrt(z * z - 1.0)) / torch.sqrt(curv)
    d = torch.log(first_term + torch.sqrt(first_term_square - 1.0)) / torch.sqrt(curv)
    return 0.5 * chord + 0.5 * d

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def pairwise_angle_at_vertex(v: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6, apply_cos=True) -> Tensor:
    """
    Given two vectors `v` and `y` on the hyperboloid, compute the angle at `v`
    in the hyperbolic triangle `OVY` where `O` is the origin of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """
    sqrt_curv = curv.sqrt()
    OV = torch.acosh(sqrt_curv * time_component(v, curv))[:,None] / sqrt_curv
    OY = torch.acosh(sqrt_curv * time_component(y, curv))[None] / sqrt_curv
    VY = pairwise_dist(v, y, curv, eps)

    cos_V = (torch.cosh(OV) * torch.cosh(VY) - torch.cosh(OY)) / (torch.sinh(OV) * torch.sinh(VY) + eps)

    if apply_cos:
        return torch.acos(torch.clamp(cos_V, min=-1, max=1))
    else:
        return cos_V

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def pairwise_angle_at_origin(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6, apply_cos=True):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the angle at the
    origin `O` in the hyperbolic triangle `Oxy` where `O` is the origin of
    the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """
    sqrt_curv = curv.sqrt()
    OX = torch.acosh(sqrt_curv * time_component(x, curv))[:,None] / sqrt_curv
    OY = torch.acosh(sqrt_curv * time_component(y, curv))[None] / sqrt_curv
    XY = pairwise_dist(x, y, curv, eps)

    cos_O = (torch.cosh(OX) * torch.cosh(OY) - torch.cosh(XY)) / (torch.sinh(OX) * torch.sinh(OY) + eps)
    
    if apply_cos:
        return torch.acos(torch.clamp(cos_O, min=-1, max=1))
    else:
        return cos_O
    
@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def elementwise_angle_at_origin(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6, apply_cos=True):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the angle at the
    origin `O` in the hyperbolic triangle `Oxy` where `O` is the origin of
    the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """
    sqrt_curv = curv.sqrt()
    OX = torch.acosh(sqrt_curv * time_component(x, curv)) / sqrt_curv
    OY = torch.acosh(sqrt_curv * time_component(y, curv)) / sqrt_curv
    XY = elementwise_dist(x, y, curv, eps)

    cos_O = (torch.cosh(OX) * torch.cosh(OY) - torch.cosh(XY)) / (torch.sinh(OX) * torch.sinh(OY) + eps)
    
    if apply_cos:
        return torch.acos(torch.clamp(cos_O, min=-1, max=1))
    else:
        return cos_O
    
@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def lorentz_to_klein(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Convert points from the Lorentz model to the Klein model of hyperbolic geometry.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors in the Klein model.
    """
    x_time = time_component(x, curv).unsqueeze(-1)
    _output = x / torch.clamp(x_time, min=eps)
    return _output

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def klein_to_lorentz(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Convert points from the Klein model to the Lorentz model of hyperbolic geometry.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            in the Klein model.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """
    x_norm_2 = torch.sum(x.square(), dim=-1, keepdim=True)
    _output = x / torch.clamp(torch.sqrt(curv * (1 - x_norm_2)), min=eps)
    return _output

@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def lorentz_factor(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Compute the Lorentz factor (gamma) of points on the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B, )` giving the Lorentz factor of the input points.
    """
    # x must be in Klein coordinates
    denom = torch.sqrt(1 - curv * torch.sum(x.square(), dim=-1, keepdim=True))
    gamma = 1 / torch.clamp(denom, min=eps)
    return gamma
       
@torch.autocast(device_type=_device_type, dtype=_cast_dtype, enabled=_enable_autocast)
def einstein_midpoint(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Compute the Einstein midpoint of two points on the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of shape `(B, D)` giving space components of the Einstein
        midpoint between input points.
    """
    x_klein = lorentz_to_klein(x, curv, eps)
    lorentz_factors = lorentz_factor(x_klein, curv, eps)
    num = torch.sum(lorentz_factors * x_klein, dim=0, keepdim=True)
    denom = torch.sum(lorentz_factors, dim=0, keepdim=True)
    midpoint_klein = num / torch.clamp(denom, min=eps)
    midpoint_lorentz = klein_to_lorentz(midpoint_klein, curv, eps)
    return midpoint_lorentz

def lorentz_to_poincare(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Convert points from the Lorentz model to the Poincare model of hyperbolic geometry.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.
    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors in the Poincare model.
    """
    x_time = time_component(x, curv).unsqueeze(-1)
    sqrt_curv = curv.sqrt()
    x_time = sqrt_curv * x_time
    _output = x / torch.clamp(x_time + 1.0, min=eps)
    return _output

def poincare_to_lorentz(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Convert points from the Poincare model to the Lorentz model of hyperbolic geometry.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            in the Poincare model.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.
    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """
    x_norm_2 = torch.sum(x.square(), dim=-1, keepdim=True)
    _output = (2 * x) / torch.clamp(1 - curv * x_norm_2, min=eps)
    return _output


##### DICTIONARY OF FUNCTIONS #####
hyperbolic_metric_functions = {
    'pairwise_inner': pairwise_inner,
    'pairwise_sim': pairwise_sim,
    'pairwise_dist': pairwise_dist,
    'cosine_similarity': cosine_similarity
}