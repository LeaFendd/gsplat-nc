"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import os
import math

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(device=device)
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("calc_compensations", [False, True])
def test_projection(test_data, calc_compensations: bool):
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from gsplat.cuda._wrapper import fully_fused_projection, quat_scale_to_covar_preci

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    # forward
    radii, means2d, depths, conics, normals, compensations = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=calc_compensations,
    )

    _covars, _ = quat_scale_to_covar_preci(quats, scales, triu=False)  # [N, 3, 3]
    _radii, _means2d, _depths, _conics, _compensations = _fully_fused_projection(
        means,
        _covars,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=calc_compensations,
    )

    # radii is integer so we allow for 1 unit difference
    valid = (radii > 0) & (_radii > 0)
    torch.testing.assert_close(radii, _radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conics[valid], _conics[valid], rtol=1e-4, atol=1e-4)
    if calc_compensations:
        torch.testing.assert_close(
            compensations[valid], _compensations[valid], rtol=1e-4, atol=1e-3
        )

    # backward
    v_means2d = torch.randn_like(means2d) * radii[..., None]
    v_depths = torch.randn_like(depths) * radii
    v_conics = torch.randn_like(conics) * radii[..., None]
    v_normals = torch.randn_like(normals) * radii[..., None]
    if calc_compensations:
        v_compensations = torch.randn_like(compensations) * radii

    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (conics * v_conics).sum()
        + (normals * v_normals).sum()
        + ((compensations * v_compensations).sum() if calc_compensations else 0),
        (viewmats, quats, scales, means),
    )
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum()
        + ((_compensations * v_compensations).sum() if calc_compensations else 0),
        (viewmats, quats, scales, means),
    )

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-3, atol=1e-3)
    # torch.testing.assert_close(v_quats, _v_quats, rtol=2e-1, atol=1e-2)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3, 32, 128])
def test_rasterize_to_pixels(test_data, channels: int):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    from gsplat.cuda._torch_impl import _rasterize_to_pixels
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        persp_proj,
        quat_scale_to_covar_preci,
        rasterize_to_pixels,
    )

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    means = test_data["means"]
    opacities = test_data["opacities"]
    C = len(Ks)
    colors = torch.randn(C, len(means), channels, device=device)
    backgrounds = torch.rand((C, colors.shape[-1]), device=device)

    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    # Project Gaussians to 2D
    radii, means2d, depths, conics, normals, compensations = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, width, height
    )
    opacities = opacities.repeat(C, 1)

    # Identify intersecting tiles
    tile_size = 16 if channels <= 32 else 4
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    means2d.requires_grad = True
    conics.requires_grad = True
    normals.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True

    # forward
    render_colors, render_alphas, render_normals, render_norm_uc = rasterize_to_pixels(
        means2d,
        conics,
        normals,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )
    _render_colors, _render_alphas, _render_normals = _rasterize_to_pixels(
        means2d,
        conics,
        normals,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )
    torch.testing.assert_close(render_colors, _render_colors)
    torch.testing.assert_close(render_alphas, _render_alphas)
    torch.testing.assert_close(render_normals, _render_normals)

    # backward
    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)
    v_render_normals = torch.randn_like(render_normals)
    v_render_norm_uc = torch.randn_like(render_norm_uc)

    v_means2d, v_conics, v_normals, v_colors, v_opacities, v_backgrounds = (
        torch.autograd.grad(
            (render_colors * v_render_colors).sum()
            + (render_alphas * v_render_alphas).sum()
            + (render_norm_uc * v_render_norm_uc).sum()
            + (render_normals * v_render_normals).sum(),
            (means2d, conics, normals, colors, opacities, backgrounds),
        )
    )
    (
        _v_means2d,
        _v_conics,
        _v_colors,
        _v_opacities,
        _v_backgrounds,
    ) = torch.autograd.grad(
        (_render_colors * v_render_colors).sum()
        + (_render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )
    torch.testing.assert_close(v_means2d, _v_means2d, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(v_conics, _v_conics, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_opacities, _v_opacities, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-3, atol=1e-3)
