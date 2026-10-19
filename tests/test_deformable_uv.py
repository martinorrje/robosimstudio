# SPDX-License-Identifier: Apache-2.0
"""Authored garment maps must survive deformation and reject broken topology."""

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.render.uv import deformable_uvs, planar_uvs

TIER = "fast"


class TestDeformableUvs(unittest.TestCase):
    def test_authored_coordinates_do_not_follow_the_current_shape(self):
        uv = np.array([[0.1, 0.2], [0.7, 0.4], [0.2, 0.9]], dtype=np.float32)
        handle = SimpleNamespace(uvs=uv)
        for points in (np.eye(3), np.eye(3) * 2 + 5):
            np.testing.assert_array_equal(deformable_uvs(handle, points), uv)

    def test_untextured_surfaces_keep_the_existing_projection(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(deformable_uvs(SimpleNamespace(), points), planar_uvs(points))

    def test_mismatched_or_nonfinite_coordinates_fail(self):
        for uv in (np.zeros((2, 2)), np.full((3, 2), np.nan)):
            with self.assertRaises(ValueError):
                deformable_uvs(SimpleNamespace(uvs=uv), np.eye(3))
