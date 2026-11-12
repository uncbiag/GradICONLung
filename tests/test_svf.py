import math
import unittest

import itk
import numpy as np
import torch
from icon_registration.itk_wrapper import resampling_transform

from gradiconlung import svf

SPACING = (2.0, 1.5, 1.0)


def smooth_field(amplitude=1.0, shape=(12, 14, 16)):
    z, y, x = torch.meshgrid(*[torch.linspace(0, math.pi, n) for n in shape], indexing="ij")
    return amplitude * torch.stack([torch.sin(y), torch.sin(x) * torch.cos(z), torch.sin(z)])[None]


def oblique_image():
    """An anisotropic image with a rotated, flipped direction and a non-zero origin."""
    image = itk.image_from_array(np.random.rand(30, 40, 50).astype(np.float32))
    image.SetSpacing((1.5, 2.0, 3.0))
    image.SetOrigin((7.0, -11.0, 5.0))
    direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([-1.0, 1.0, 1.0])
    image.SetDirection(itk.matrix_from_array(direction))
    return image


class TestSVF(unittest.TestCase):
    def test_exp_of_zero_is_identity(self):
        v = torch.zeros(1, 3, 6, 7, 8)
        self.assertTrue(torch.equal(svf.exp(v, SPACING), v))

    def test_exp_of_constant_field_is_translation(self):
        v = torch.ones(1, 3, 6, 7, 8) * torch.tensor([0.3, -0.2, 0.1]).view(1, 3, 1, 1, 1)
        self.assertTrue(torch.allclose(svf.exp(v, SPACING), v, atol=1e-6))

    def test_exp_inverse(self):
        v = smooth_field(0.5)
        forward, backward = svf.exp(v, SPACING), svf.exp(-v, SPACING)
        # exp(v) o exp(-v) = Id  <=>  backward + forward(Id + backward) = 0
        composed = backward + svf.warp(forward, backward, SPACING)
        self.assertLess(composed[:, :, 2:-2, 2:-2, 2:-2].abs().max(), 0.02)

    def test_fit_improves_on_its_initialization(self):
        # Large enough that the starting guess 0.5 * (forward - backward) is clearly off.
        v = smooth_field(4.0)
        forward, backward = svf.exp(v, SPACING), svf.exp(-v, SPACING)
        initial_error = (0.5 * (forward - backward) - v).abs().mean()
        error = (svf.fit_svf(forward, backward, SPACING) - v).abs().mean()
        self.assertLess(error, initial_error / 3)
        self.assertLess(error, 0.05)

    def test_velocity_image_lies_on_the_network_grid(self):
        image = oblique_image()
        shape = (20, 22, 24)
        velocity = svf.velocity_image(torch.zeros(1, 3, *shape), image)
        network_to_image = resampling_transform(image, list(reversed(shape)))
        for index in [(0, 0, 0), (23, 21, 19), (5, 7, 11)]:
            np.testing.assert_allclose(
                velocity.TransformIndexToPhysicalPoint(index), network_to_image.TransformPoint(index), atol=1e-6
            )

    def test_velocity_image_matches_the_transform(self):
        # For a constant v, exp(v) = v, so the transform must move every point by the saved vector.
        image = oblique_image()
        shape = (20, 22, 24)
        v_mm = torch.tensor([6.0, -4.0, 3.0])
        v = torch.ones(1, 3, *shape) * v_mm.view(1, 3, 1, 1, 1)
        spacing = svf.network_spacing(image, shape)
        transform = svf.itk_transform(v, spacing, image, image)
        velocity = svf.velocity_image(v, image)
        vectors = itk.array_from_image(velocity)
        for index in [(12, 11, 10), (6, 9, 13), (17, 14, 8)]:
            point = np.array(velocity.TransformIndexToPhysicalPoint(index))
            np.testing.assert_allclose(
                np.array(transform.TransformPoint(tuple(point))) - point, vectors[index[::-1]], atol=1e-3
            )
        self.assertAlmostEqual(np.linalg.norm(vectors[10, 11, 12]), v_mm.norm().item(), places=4)

    def test_same_physical_box(self):
        image = oblique_image()
        self.assertTrue(svf.same_physical_box(image, image))
        shifted = oblique_image()
        shifted.SetOrigin((8.0, -11.0, 5.0))
        self.assertFalse(svf.same_physical_box(image, shifted))


if __name__ == "__main__":
    unittest.main()
