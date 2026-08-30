"""Tests for the geometry-only fallback used by the policy overlay."""

import numpy as np

from humanoid_climber.terrain_probe import terrain_metrics_from_height_grid


def test_flat_height_grid_is_classified_flat() -> None:
    estimate = terrain_metrics_from_height_grid(
        np.zeros((5, 5)), spacing_m=0.25
    )
    assert estimate.terrain_label == "flat"
    assert estimate.slope_gradient is not None
    assert estimate.slope_gradient < 1.0e-8
    assert estimate.roughness_m == 0.0


def test_planar_height_grid_recovers_gradient() -> None:
    x = np.linspace(-0.5, 0.5, 5)
    heights = np.tile(0.18 * x, (5, 1))
    estimate = terrain_metrics_from_height_grid(heights, spacing_m=0.25)
    assert estimate.terrain_label == "incline"
    assert estimate.slope_gradient is not None
    assert abs(estimate.slope_gradient - 0.18) < 1.0e-6
    assert estimate.roughness_m is not None
    assert estimate.roughness_m < 1.0e-8


def test_smooth_pyramid_transition_is_incline_not_rough() -> None:
    coords = np.linspace(-0.8, 0.8, 9)
    xx, yy = np.meshgrid(coords, coords)
    radial = np.maximum(np.abs(xx), np.abs(yy))
    heights = -0.20 * np.maximum(radial - 0.4, 0.0)
    estimate = terrain_metrics_from_height_grid(heights, spacing_m=0.2)
    assert estimate.terrain_label == "incline"
    assert estimate.slope_gradient is not None
    assert estimate.slope_gradient > 0.06
    assert estimate.roughness_m is not None
    assert estimate.roughness_m < 0.03


def test_high_frequency_heightfield_relief_is_rough() -> None:
    row, col = np.indices((9, 9))
    heights = ((row + col) % 2).astype(float) * 0.06
    estimate = terrain_metrics_from_height_grid(heights, spacing_m=0.2)
    assert estimate.terrain_label == "rough terrain"
    assert estimate.roughness_m is not None
    assert estimate.roughness_m > 0.03


def test_box_height_discontinuity_is_classified_as_steps() -> None:
    heights = np.zeros((5, 5))
    heights[:, 2:] = 0.12
    estimate = terrain_metrics_from_height_grid(
        heights, spacing_m=0.25, box_dominated=True
    )
    assert estimate.terrain_label == "stairs / obstacles"
    assert estimate.step_height_m is not None
    assert estimate.step_height_m > 0.05
