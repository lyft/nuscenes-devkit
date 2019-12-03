# Lyft dataset SDK
# based on the code written by Alex Lang and Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

"""
https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/discussion/112409649874


This script converts nuScenes data to KITTI format and KITTI results to nuScenes.
It is used for compatibility with software that uses KITTI-style annotations.

The difference beteeen formats:
    KITTI has only front-facing cameras, whereas nuScenes has a 360 degree horizontal fov.
    KITTI has no radar data.
    The nuScenes database format is more modular.
    KITTI fields like occluded and truncated cannot be exactly reproduced from nuScenes data.
    KITTI has different categories.

Current limitations of the script.:
    We don't specify the KITTI imu_to_velo_kitti projection in this code base.
    We map nuScenes categories to nuScenes detection categories, rather than KITTI categories.
    Attributes are not part of KITTI and therefore set to '' in the nuScenes result format.
    Velocities are not part of KITTI and therefore set to 0 in the nuScenes result format.
    This script uses the train and val splits of nuScenes, whereas standard KITTI has training and testing splits.
"""

from pathlib import Path
from typing import List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, transform_matrix
from lyft_dataset_sdk.utils.kitti import KittiDB
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm


class KittiConverter:
    def __init__(self, store_dir: str = "~/lyft_kitti/train/"):
        """

        Args:
            store_dir: Where to write the KITTI-style annotations.
        """
        self.store_dir = Path(store_dir).expanduser()

        # Create store_dir.
        if not self.store_dir.is_dir():
            self.store_dir.mkdir(parents=True)

    def nuscenes_gt_to_kitti(
        self,
        lyft_dataroot: str,
        table_folder: str,
        lidar_name: str = "LIDAR_TOP",
        get_all_detections: bool = False,
        parallel_n_jobs: int = 4,
        samples_count: Optional[int] = None,
    ) -> None:
        """Converts nuScenes GT formatted annotations to KITTI format.

        Args:
            lyft_dataroot: folder with tables (json files).
            table_folder: folder with tables (json files).
            lidar_name: Name of the lidar sensor.
                Only one lidar allowed at this moment.
            get_all_detections: If True, will write all
                bboxes in PointCloud and use only FrontCamera.
            parallel_n_jobs: Number of threads to parralel processing.
            samples_count: Number of samples to convert.

        """
        self.lyft_dataroot = lyft_dataroot
        self.table_folder = table_folder
        self.lidar_name = lidar_name
        self.get_all_detections = get_all_detections
        self.samples_count = samples_count
        self.parallel_n_jobs = parallel_n_jobs

        # Select subset of the data to look at.
        self.lyft_ds = LyftDataset(self.lyft_dataroot, self.table_folder)

        self.kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi)
        self.kitti_to_nu_lidar_inv = self.kitti_to_nu_lidar.inverse

        # Get assignment of scenes to splits.
        split_logs = [self.lyft_ds.get("log", scene["log_token"])["logfile"] for scene in self.lyft_ds.scene]
        if self.get_all_detections:
            self.cams_to_see = ["CAM_FRONT"]
        else:
            self.cams_to_see = [
                "CAM_FRONT",
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]

        # Create output folders.
        self.label_folder = self.store_dir.joinpath("label_2")
        self.calib_folder = self.store_dir.joinpath("calib")
        self.image_folder = self.store_dir.joinpath("image_2")
        self.lidar_folder = self.store_dir.joinpath("velodyne")
        for folder in [self.label_folder, self.calib_folder, self.image_folder, self.lidar_folder]:
            if not folder.is_dir():
                folder.mkdir(parents=True)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        if self.samples_count is not None:
            sample_tokens = sample_tokens[: self.samples_count]

        with parallel_backend("threading", n_jobs=self.parallel_n_jobs):
            Parallel()(delayed(self.process_token_to_kitti)(sample_token) for sample_token in tqdm(sample_tokens))

    def process_token_to_kitti(self, sample_token: str) -> None:
        # Get sample data.
        sample = self.lyft_ds.get("sample", sample_token)
        sample_annotation_tokens = sample["anns"]

        lidar_token = sample["data"][self.lidar_name]
        sd_record_lid = self.lyft_ds.get("sample_data", lidar_token)
        cs_record_lid = self.lyft_ds.get("calibrated_sensor", sd_record_lid["calibrated_sensor_token"])
        ego_record_lid = self.lyft_ds.get("ego_pose", sd_record_lid["ego_pose_token"])
        for cam_name in self.cams_to_see:
            cam_front_token = sample["data"][cam_name]
            if self.get_all_detections:
                token_to_write = sample_token
            else:
                token_to_write = cam_front_token

            # Retrieve sensor records.
            sd_record_cam = self.lyft_ds.get("sample_data", cam_front_token)
            cs_record_cam = self.lyft_ds.get("calibrated_sensor", sd_record_cam["calibrated_sensor_token"])
            ego_record_cam = self.lyft_ds.get("ego_pose", sd_record_cam["ego_pose_token"])
            cam_height = sd_record_cam["height"]
            cam_width = sd_record_cam["width"]
            imsize = (cam_width, cam_height)

            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(
                cs_record_lid["translation"], Quaternion(cs_record_lid["rotation"]), inverse=False
            )
            lid_ego_to_world = transform_matrix(
                ego_record_lid["translation"], Quaternion(ego_record_lid["rotation"]), inverse=False
            )
            world_to_cam_ego = transform_matrix(
                ego_record_cam["translation"], Quaternion(ego_record_cam["rotation"]), inverse=True
            )
            ego_to_cam = transform_matrix(
                cs_record_cam["translation"], Quaternion(cs_record_cam["rotation"]), inverse=True
            )
            velo_to_cam = np.dot(ego_to_cam, np.dot(world_to_cam_ego, np.dot(lid_ego_to_world, lid_to_ego)))

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(velo_to_cam, self.kitti_to_nu_lidar.transformation_matrix)

            # Currently not used.
            imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            # Cameras are always rectified.
            p_left_kitti[:3, :3] = cs_record_cam["camera_intrinsic"]

            # Create KITTI style transforms.
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

            # Check that the rotation has the same format as in KITTI.
            if self.lyft_ds.get("sensor", cs_record_cam["sensor_token"])["channel"] == "CAM_FRONT":
                expected_kitti_velo_to_cam_rot = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
                assert (velo_to_cam_rot.round(0) == expected_kitti_velo_to_cam_rot).all(), velo_to_cam_rot.round(0)

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will
            # include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam["filename"]
            filename_lid_full = sd_record_lid["filename"]

            # Convert image (jpg to png).
            src_im_path = self.lyft_ds.data_path.joinpath(filename_cam_full)
            dst_im_path = self.image_folder.joinpath(f"{token_to_write}.png")
            if not dst_im_path.exists():
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = self.lyft_ds.data_path.joinpath(filename_lid_full)
            dst_lid_path = self.lidar_folder.joinpath(f"{token_to_write}.bin")

            pcl = LidarPointCloud.from_file(Path(src_lid_path))
            # In KITTI lidar frame.
            pcl.rotate(self.kitti_to_nu_lidar_inv.rotation_matrix)
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            # Add to tokens.
            # tokens.append(token_to_write)

            # Create calibration file.
            kitti_transforms = dict()
            kitti_transforms["P0"] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms["P1"] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms["P2"] = p_left_kitti  # Left camera transform.
            kitti_transforms["P3"] = np.zeros((3, 4))  # Dummy values.
            # Cameras are already rectified.
            kitti_transforms["R0_rect"] = r0_rect.rotation_matrix
            kitti_transforms["Tr_velo_to_cam"] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            kitti_transforms["Tr_imu_to_velo"] = imu_to_velo_kitti
            calib_path = self.calib_folder.joinpath(f"{token_to_write}.txt")

            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = "%.12e" % val[0]
                    for v in val[1:]:
                        val_str += " %.12e" % v
                    calib_file.write("%s: %s\n" % (key, val_str))

            # Write label file.
            label_path = self.label_folder.joinpath(f"{token_to_write}.txt")
            if label_path.exists():
                print("Skipping existing file: %s" % label_path)
                continue
            with open(label_path, "w") as label_file:
                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = self.lyft_ds.get("sample_annotation", sample_annotation_token)

                    # Get box in LIDAR frame.
                    _, box_lidar_nusc, _ = self.lyft_ds.get_sample_data(
                        lidar_token, box_vis_level=BoxVisibility.NONE, selected_anntokens=[sample_annotation_token]
                    )
                    box_lidar_nusc = box_lidar_nusc[0]

                    # Truncated: Set all objects to 0 which means untruncated.
                    truncated = 0.0

                    # Occluded: Set all objects to full visibility as this information is
                    # not available in nuScenes.
                    occluded = 0

                    detection_name = sample_annotation["category_name"]

                    # Convert from nuScenes to KITTI box format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect
                    )

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
                    if bbox_2d is None and not self.get_all_detections:
                        continue
                    elif bbox_2d is None and self.get_all_detections:
                        # default KITTI bbox
                        bbox_2d = (-1.0, -1.0, -1.0, -1.0)

                    # Set dummy score so we can use this file as result.
                    box_cam_kitti.score = 0

                    # Convert box to output string format.
                    output = KittiDB.box_to_string(
                        name=detection_name,
                        box=box_cam_kitti,
                        bbox_2d=bbox_2d,
                        truncation=truncated,
                        occlusion=occluded,
                    )

                    # Write to disk.
                    label_file.write(output + "\n")

    def render_kitti(self, render_2d: bool = False) -> None:
        """Renders the annotations in the KITTI dataset from a lidar and a camera view.

        Args:
            render_2d: Whether to render 2d boxes (only works for camera data).

        Returns:

        """
        if render_2d:
            print("Rendering 2d boxes from KITTI format")
        else:
            print("Rendering 3d boxes projected from 3d KITTI format")

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.store_dir)

        # Create output folder.
        render_dir = self.store_dir.joinpath("render")
        if not render_dir.is_dir():
            render_dir.mkdir(parents=True)

        # Render each image.
        tokens = kitti.tokens

        # currently supports only single thread processing
        for token in tqdm(tokens):

            for sensor in ["lidar", "camera"]:
                out_path = render_dir.joinpath(f"{token}_{sensor}.png")
                kitti.render_sample_data(token, sensor_modality=sensor, out_path=out_path, render_2d=render_2d)
                # Close the windows to avoid a warning of too many open windows.
                plt.close()

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """Convenience function to get the samples in a particular split.

        Args:
            split_logs: A list of the log names in this split.

        Returns: The list of samples.

        """
        samples = []
        for sample in self.lyft_ds.sample:
            scene = self.lyft_ds.get("scene", sample["scene_token"])
            log = self.lyft_ds.get("log", scene["log_token"])
            logfile = log["logfile"]
            if logfile in split_logs:
                samples.append(sample["token"])
        return samples


if __name__ == "__main__":
    fire.Fire(KittiConverter)
