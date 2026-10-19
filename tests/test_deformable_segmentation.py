# SPDX-License-Identifier: Apache-2.0
"""Cloth hits are reserved ray IDs, never rigid-shape array indices."""

import unittest

import numpy as np

from robosimstudio.scene.handles import DeformableHandle, SceneHandles
from robosimstudio.solvers.render.tiled import RendererTiled

TIER = "fast"


class TestDeformableSegmentation(unittest.TestCase):
    def test_single_cloth_has_a_named_mask_without_renumbering_the_robot(self):
        handles = SceneHandles(deformables={"sock": DeformableHandle(name="sock")})
        self.assertEqual(handles.segmentation_names(), {1: "__robot__", 2: "sock"})
        self.assertEqual(handles.segmentation_labels(), {0xFFFFFFFD: 2, 0xFFFFFFFE: 2})

    def test_multiple_cloths_are_not_falsely_identified_as_one(self):
        handles = SceneHandles(deformables={n: DeformableHandle(name=n) for n in ("a", "b")})
        self.assertNotIn(0xFFFFFFFD, handles.segmentation_labels())

    def test_reserved_and_unknown_ids_never_index_the_rigid_lookup(self):
        renderer = RendererTiled.__new__(RendererTiled)
        renderer._label_lut = np.array([7, 8, 0], dtype=np.int32)
        renderer._special_labels = {0xFFFFFFFD: 9, 0xFFFFFFFE: 9}
        ids = np.array([0, 1, 0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFC, -1], dtype=np.int64)
        np.testing.assert_array_equal(renderer._label_shape_indices(ids), [7, 8, 9, 9, 0, 0, 0])
