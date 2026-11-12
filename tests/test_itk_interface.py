import itk
import numpy as np
import unittest
import torch
import torch.nn.functional as F

import icon_registration.test_utils

from gradiconlung import preprocess, get_gradiconlung, register_pair, register_pair_svf


def mean_tre(transform, fixed_points, moving_points):
    """Mean distance between moving landmarks and fixed landmarks mapped by ``transform``."""
    return np.mean([
        np.linalg.norm(moving_points[i] - np.array(transform.TransformPoint(tuple(fixed_points[i]))))
        for i in range(len(fixed_points))
    ])


class TestItkInterface(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        icon_registration.test_utils.download_test_data()
        self.test_data_dir = icon_registration.test_utils.TEST_DATA_DIR
        lung = self.test_data_dir / "lung_test_data"
        self.image_exp = itk.imread(str(lung / "copd1_highres_EXP_STD_COPD_img.nii.gz"))
        self.image_insp = itk.imread(str(lung / "copd1_highres_INSP_STD_COPD_img.nii.gz"))
        self.insp_points = icon_registration.test_utils.read_copd_pointset(str(lung / "copd1_300_iBH_xyz_r1.txt"))
        self.exp_points = icon_registration.test_utils.read_copd_pointset(str(lung / "copd1_300_eBH_xyz_r1.txt"))

    def test_preprocessing_ct(self):
        img_preprocessed = preprocess(self.image_exp)

        img = torch.Tensor(np.array(self.image_exp, dtype=np.float32))
        reference = ((torch.clip(img, -1000, 1000) + 1000) / 1000).numpy()

        self.assertLess(np.mean(np.abs(reference - img_preprocessed)), 1e-5)

    def test_itk_registration(self):
        net = get_gradiconlung()
        phi_AB, phi_BA = register_pair(
            net, preprocess(self.image_insp), preprocess(self.image_exp), finetune_steps=None
        )

        assert isinstance(phi_AB, itk.CompositeTransform)
        self.assertLess(mean_tre(phi_AB, self.exp_points, self.insp_points), 2.0)
        self.assertLess(mean_tre(phi_BA, self.insp_points, self.exp_points), 2.0)

    def test_itk_registration_svf(self):
        net = get_gradiconlung()
        phi_AB, phi_BA, velocity = register_pair_svf(
            net, preprocess(self.image_insp), preprocess(self.image_exp), finetune_steps=None
        )

        self.assertEqual(itk.array_from_image(velocity).shape, (175, 175, 175, 3))
        self.assertLess(mean_tre(phi_AB, self.exp_points, self.insp_points), 2.0)
        self.assertLess(mean_tre(phi_BA, self.insp_points, self.exp_points), 2.0)

        # exp(v) and exp(-v) are inverses of each other.
        errors = [
            np.linalg.norm(np.array(phi_BA.TransformPoint(phi_AB.TransformPoint(tuple(p)))) - p)
            for p in self.exp_points
        ]
        self.assertLess(np.mean(errors), 0.25)

    def test_itk_warp(self):
        net = get_gradiconlung()
        moving, fixed = preprocess(self.image_insp), preprocess(self.image_exp)
        phi_AB, phi_BA = register_pair(net, moving, fixed, finetune_steps=None)

        interpolator = itk.LinearInterpolateImageFunction.New(moving)
        warped_moving_image = np.array(itk.resample_image_filter(
                moving,
                transform=phi_AB,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=fixed
                ))

        reference = F.interpolate(net.warped_image_A, size=warped_moving_image.shape, mode='trilinear', align_corners=False)[0,0].cpu().numpy()

        from icon_registration.losses import NCC
        diff = NCC()(torch.Tensor(warped_moving_image).unsqueeze(0).unsqueeze(0), torch.Tensor(reference).unsqueeze(0).unsqueeze(0))
        self.assertLess(
            diff, 1e-2
        )
