import itk
import numpy as np
import unittest
import icon_registration.test_utils

import subprocess
import os


class TestCommandInterface(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        icon_registration.test_utils.download_test_data()
        self.test_data_dir = icon_registration.test_utils.TEST_DATA_DIR
        self.test_temp_dir = f"{self.test_data_dir}/temp"
        os.makedirs(self.test_temp_dir, exist_ok=True)
        lung = f"{self.test_data_dir}/lung_test_data"
        self.register = [
            "gradiconlung-register",
            "--fixed", f"{lung}/copd1_highres_EXP_STD_COPD_img.nii.gz",
            "--moving", f"{lung}/copd1_highres_INSP_STD_COPD_img.nii.gz",
            "--io_iterations", "None",
        ]
        self.insp_points = icon_registration.test_utils.read_copd_pointset(f"{lung}/copd1_300_iBH_xyz_r1.txt")
        self.exp_points = icon_registration.test_utils.read_copd_pointset(f"{lung}/copd1_300_eBH_xyz_r1.txt")

    def mean_tre(self, transform_file):
        phi_AB = itk.transformread(transform_file)[0]
        assert isinstance(phi_AB, itk.CompositeTransform)
        return np.mean([
            np.linalg.norm(self.insp_points[i] - np.array(phi_AB.TransformPoint(tuple(self.exp_points[i]))))
            for i in range(len(self.exp_points))
        ])

    def test_register_inference(self):
        transform = f"{self.test_temp_dir}/transform.hdf5"
        subprocess.run(self.register + ["--transform_out", transform], check=True)

        self.assertLess(self.mean_tre(transform), 2.0)
        os.remove(transform)

    def test_register_svf(self):
        transform = f"{self.test_temp_dir}/transform_svf.hdf5"
        velocity = f"{self.test_temp_dir}/svf.nii.gz"
        subprocess.run(self.register + ["--transform_out", transform, "--svf_out", velocity], check=True)

        self.assertLess(self.mean_tre(transform), 2.0)
        self.assertEqual(itk.array_from_image(itk.imread(velocity)).shape, (175, 175, 175, 3))
        os.remove(transform)
        os.remove(velocity)
