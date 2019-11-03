# Lyft dataset SDK
# based on the code written by Alex Lang and Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

import os
from pathlib import Path
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import Box, LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points
from matplotlib.axes import Axes
from PIL import Image
from pyquaternion import Quaternion


class KittiDB:
    def __init__(self, root: Path):
        """

        Args:
            root: Base folder for all KITTI data.
        """
        self.root = root
        self.tables = ("calib", "image_2", "label_2", "velodyne")
        self._kitti_fileext = {"calib": "txt", "image_2": "png", "label_2": "txt", "velodyne": "bin"}

        # Grab all the expected tokens.
        self._kitti_tokens = []

        split_dir = self.root / "image_2"
        _tokens = os.listdir(split_dir)
        _tokens = [t.replace(".png", "") for t in _tokens]
        _tokens.sort()
        self.tokens = _tokens

        # KITTI LIDAR has the x-axis pointing forward, but our LIDAR points backwards. So we need to apply a
        # 180 degree rotation around to yaw (z-axis) in order to align.
        # The quaternions will be used a lot of time. We store them as instance variables so that we don't have
        # to create a new one every single time.
        self.kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi)
        self.kitti_to_nu_lidar_inv = self.kitti_to_nu_lidar.inverse

    @staticmethod
    def parse_label_line(label_line) -> dict:
        """Parses single line from label file into a dict. Boxes are in camera frame. See KITTI devkit for details and
        http://www.cvlibs.net/datasets/kitti/setup.php for visualizations of the setup.

        Args:
            label_line: Single line from KittiDB label file.

        Returns: Dictionary with all the line details.

        """
        parts = label_line.split(" ")
        output = {
            "name": parts[0].strip(),
            "xyz_camera": (float(parts[11]), float(parts[12]), float(parts[13])),
            "wlh": (float(parts[9]), float(parts[10]), float(parts[8])),
            "yaw_camera": float(parts[14]),
            "bbox_camera": (float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
            "truncation": float(parts[1]),
            "occlusion": float(parts[2]),
            "alpha": float(parts[3]),
        }

        # Add score if specified
        if len(parts) > 15:
            output["score"] = float(parts[15])
        else:
            output["score"] = np.nan

        return output

    @staticmethod
    def box_nuscenes_to_kitti(
        box: Box,
        velo_to_cam_rot: Quaternion,
        velo_to_cam_trans: np.ndarray,
        r0_rect: Quaternion,
        kitti_to_nu_lidar_inv: Quaternion = Quaternion(axis=(0, 0, 1), angle=np.pi).inverse,
    ) -> Box:
        """Transform from nuScenes lidar frame to KITTI reference frame.

        Args:
            box: Instance in nuScenes lidar frame.
            velo_to_cam_rot: Quaternion to rotate from lidar to camera frame.
            velo_to_cam_trans: <np.float: 3>. Translate from lidar to camera frame.
            r0_rect: Quaternion to rectify camera frame.
            kitti_to_nu_lidar_inv: Quaternion to rotate nuScenes to KITTI LIDAR.

        Returns: Box instance in KITTI reference frame.

        """
        # Copy box to avoid side-effects.
        box = box.copy()

        # Rotate to KITTI lidar.
        box.rotate(kitti_to_nu_lidar_inv)

        # Transform to KITTI camera.
        box.rotate(velo_to_cam_rot)
        box.translate(velo_to_cam_trans)

        # Rotate to KITTI rectified camera.
        box.rotate(r0_rect)

        # KITTI defines the box center as the bottom center of the object.
        # We use the true center, so we need to adjust half height in y direction.
        box.translate(np.array([0, box.wlh[2] / 2, 0]))

        return box

    @staticmethod
    def project_kitti_box_to_image(
        box: Box, p_left: np.ndarray, imsize: Tuple[int, int]
    ) -> Union[None, Tuple[int, int, int, int]]:
        """Projects 3D box into KITTI image FOV.

        Args:
            box: 3D box in KITTI reference frame.
            p_left: <np.float: 3, 4>. Projection matrix.
            imsize: (width, height). Image size.

        Returns: (xmin, ymin, xmax, ymax). Bounding box in image plane or None if box is not in the image.

        """
        # Create a new box.
        box = box.copy()

        # KITTI defines the box center as the bottom center of the object.
        # We use the true center, so we need to adjust half height in negative y direction.
        box.translate(np.array([0, -box.wlh[2] / 2, 0]))

        # Check that some corners are inside the image.
        corners = np.array([corner for corner in box.corners().T if corner[2] > 0]).T
        if len(corners) == 0:
            return None

        # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
        imcorners = view_points(corners, p_left, normalize=True)[:2]
        bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))

        # Crop bbox to prevent it extending outside image.
        bbox_crop = tuple(max(0, b) for b in bbox)
        bbox_crop = (
            min(imsize[0], bbox_crop[0]),
            min(imsize[0], bbox_crop[1]),
            min(imsize[0], bbox_crop[2]),
            min(imsize[1], bbox_crop[3]),
        )

        # Detect if a cropped box is empty.
        if bbox_crop[0] >= bbox_crop[2] or bbox_crop[1] >= bbox_crop[3]:
            return None

        return bbox_crop

    @staticmethod
    def get_filepath(token: str, table: str, root: Path) -> str:
        """For a token and table, get the filepath to the associated data.

        Args:
            token: KittiDB unique id.
            table: Type of table, for example image or velodyne.
            root: Base folder for all KITTI data.

        Returns: Full get_filepath to desired data.

        """
        kitti_fileext = {"calib": "txt", "image_2": "png", "label_2": "txt", "velodyne": "bin"}

        ending = kitti_fileext[table]

        filepath = root / table / f"{token}.{ending}"

        return str(filepath)

    @staticmethod
    def get_transforms(token: str, root: Path) -> dict:
        calib_filename = KittiDB.get_filepath(token, "calib", root=root)

        lines = [line.rstrip() for line in open(calib_filename)]
        velo_to_cam = np.array(lines[5].strip().split(" ")[1:], dtype=np.float64)
        velo_to_cam.resize((3, 4))

        r0_rect = np.array(lines[4].strip().split(" ")[1:], dtype=np.float64)
        r0_rect.resize((3, 3))
        p_left = np.array(lines[2].strip().split(" ")[1:], dtype=np.float64)
        p_left.resize((3, 4))

        # Merge rectification and projection into one matrix.
        p_combined = np.eye(4)
        p_combined[:3, :3] = r0_rect
        p_combined = np.dot(p_left, p_combined)
        return {
            "velo_to_cam": {"R": velo_to_cam[:, :3], "T": velo_to_cam[:, 3]},
            "r0_rect": r0_rect,
            "p_left": p_left,
            "p_combined": p_combined,
        }

    @staticmethod
    def get_pointcloud(token: str, root: Path) -> LidarPointCloud:
        """Load up the point cloud for a sample.

        Args:
            token: KittiDB unique id.
            root: Base folder for all KITTI data.

        Returns: LidarPointCloud for the sample in the KITTI Lidar frame.

        """
        pc_filename = KittiDB.get_filepath(token, "velodyne", root=root)

        # The lidar PC is stored in the KITTI LIDAR coord system.
        pc = LidarPointCloud(np.fromfile(pc_filename, dtype=np.float32).reshape(-1, 4).T)

        return pc

    def get_boxes(self, token: str, filter_classes: List[str] = None, max_dist: float = None) -> List[Box]:
        """Load up all the boxes associated with a sample.
            Boxes are in nuScenes lidar frame.

        Args:
            token: KittiDB unique id.
            filter_classes: List of Kitti classes to use or None to use all.
            max_dist: List of Kitti classes to use or None to use all.

        Returns: Boxes in nuScenes lidar reference frame.

        """
        # Get transforms for this sample
        transforms = self.get_transforms(token, root=self.root)

        boxes = []
        if token.startswith("test_"):
            # No boxes to return for the test set.
            return boxes

        with open(KittiDB.get_filepath(token, "label_2", root=self.root), "r") as f:
            for line in f:
                # Parse this line into box information.
                parsed_line = self.parse_label_line(line)

                if parsed_line["name"] in {"DontCare", "Misc"}:
                    continue

                center = parsed_line["xyz_camera"]
                wlh = parsed_line["wlh"]
                yaw_camera = parsed_line["yaw_camera"]
                name = parsed_line["name"]
                score = parsed_line["score"]

                # Optional: Filter classes.
                if filter_classes is not None and name not in filter_classes:
                    continue

                # The Box class coord system is oriented the same way as as KITTI LIDAR: x forward, y left, z up.
                # For orientation confer: http://www.cvlibs.net/datasets/kitti/setup.php.

                # 1: Create box in Box coordinate system with center at origin.
                # The second quaternion in yaw_box transforms the coordinate frame from the object frame
                # to KITTI camera frame. The equivalent cannot be naively done afterwards, as it's a rotation
                # around the local object coordinate frame, rather than the camera frame.
                quat_box = Quaternion(axis=(0, 1, 0), angle=yaw_camera) * Quaternion(axis=(1, 0, 0), angle=np.pi / 2)
                box = Box([0.0, 0.0, 0.0], wlh, quat_box, name=name)

                # 2: Translate: KITTI defines the box center as the bottom center of the vehicle. We use true center,
                # so we need to add half height in negative y direction, (since y points downwards), to adjust. The
                # center is already given in camera coord system.
                box.translate(center + np.array([0, -wlh[2] / 2, 0]))

                # 3: Transform to KITTI LIDAR coord system. First transform from rectified camera to camera, then
                # camera to KITTI lidar.

                box.rotate(Quaternion(matrix=transforms["r0_rect"]).inverse)
                box.translate(-transforms["velo_to_cam"]["T"])
                box.rotate(Quaternion(matrix=transforms["velo_to_cam"]["R"]).inverse)
                # 4: Transform to nuScenes LIDAR coord system.
                box.rotate(self.kitti_to_nu_lidar)

                # Set score or NaN.
                box.score = score

                # Set dummy velocity.
                box.velocity = np.array((0.0, 0.0, 0.0))

                # Optional: Filter by max_dist
                if max_dist is not None:
                    dist = np.sqrt(np.sum(box.center[:2] ** 2))
                    if dist > max_dist:
                        continue

                boxes.append(box)

        return boxes

    def get_boxes_2d(
        self, token: str, filter_classes: List[str] = None
    ) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
        """Get the 2d boxes associated with a sample.

        Args:
            token:
            filter_classes:

        Returns: A list of boxes in KITTI format (xmin, ymin, xmax, ymax) and a list of the class names.

        """
        boxes = []
        names = []
        with open(KittiDB.get_filepath(token, "label_2", root=self.root), "r") as f:
            for line in f:
                # Parse this line into box information.
                parsed_line = self.parse_label_line(line)

                if parsed_line["name"] in {"DontCare", "Misc"}:
                    continue

                bbox_2d = parsed_line["bbox_camera"]
                name = parsed_line["name"]

                # Optional: Filter classes.
                if filter_classes is not None and name not in filter_classes:
                    continue

                boxes.append(bbox_2d)
                names.append(name)
        return boxes, names

    @staticmethod
    def box_to_string(
        name: str,
        box: Box,
        bbox_2d: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0),
        truncation: float = -1.0,
        occlusion: int = -1,
        alpha: float = -10.0,
    ) -> str:
        """Convert box in KITTI image frame to official label string fromat.

        Args:
            name: KITTI name of the box.
            box: Box class in KITTI image frame.
            bbox_2d: Optional, 2D bounding box obtained by projected Box into image (xmin, ymin, xmax, ymax).
                Otherwise set to KITTI default.
            truncation: Optional truncation, otherwise set to KITTI default.
            occlusion: Optional occlusion, otherwise set to KITTI default.
            alpha: Optional alpha, otherwise set to KITTI default.

        Returns: KITTI string representation of box.

        """
        # Convert quaternion to yaw angle.
        v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])

        # Prepare output.
        name += " "
        trunc = "{:.2f} ".format(truncation)
        occ = "{:d} ".format(occlusion)
        a = "{:.2f} ".format(alpha)
        bb = "{:.2f} {:.2f} {:.2f} {:.2f} ".format(bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
        hwl = "{:.2} {:.2f} {:.2f} ".format(box.wlh[2], box.wlh[0], box.wlh[1])  # height, width, length.
        xyz = "{:.2f} {:.2f} {:.2f} ".format(box.center[0], box.center[1], box.center[2])  # x, y, z.
        y = "{:.2f}".format(yaw)  # Yaw angle.
        s = " {:.4f}".format(box.score)  # Classification score.

        output = name + trunc + occ + a + bb + hwl + xyz + y
        if ~np.isnan(box.score):
            output += s

        return output

    def project_pts_to_image(self, pointcloud: LidarPointCloud, token: str) -> np.ndarray:
        """Project lidar points into image.

        Args:
            pointcloud: The LidarPointCloud in nuScenes lidar frame.
            token: Unique KITTI token.

        Returns: <np.float: N, 3.> X, Y are points in image pixel coordinates. Z is depth in image.

        """
        # Copy and convert pointcloud.
        pc_image = LidarPointCloud(points=pointcloud.points.copy())
        pc_image.rotate(self.kitti_to_nu_lidar_inv)  # Rotate to KITTI lidar.

        # Transform pointcloud to camera frame.
        transforms = self.get_transforms(token, root=self.root)
        pc_image.rotate(transforms["velo_to_cam"]["R"])
        pc_image.translate(transforms["velo_to_cam"]["T"])

        # Project to image.
        depth = pc_image.points[2, :]
        points_fov = view_points(pc_image.points[:3, :], transforms["p_combined"], normalize=True)
        points_fov[2, :] = depth

        return points_fov

    def render_sample_data(
        self,
        token: str,
        sensor_modality: str = "lidar",
        with_anns: bool = True,
        axes_limit: float = 30,
        ax: Axes = None,
        view_3d: np.ndarray = np.eye(4),
        color_func: Any = None,
        augment_previous: bool = False,
        box_linewidth: int = 2,
        filter_classes: List[str] = None,
        max_dist: float = None,
        out_path: str = None,
        render_2d: bool = False,
    ) -> None:
        """Render sample data onto axis. Visualizes lidar in nuScenes lidar frame and camera in camera frame.

        Args:
            token: KITTI token.
            sensor_modality: The modality to visualize, e.g. lidar or camera.
            with_anns: Whether to draw annotations.
            axes_limit: Axes limit for lidar data (measured in meters).
            ax: Axes onto which to render.
            view_3d: 4x4 view matrix for 3d views.
            color_func: Optional function that defines the render color given the class name.
            augment_previous: Whether to augment an existing plot (does not redraw pointcloud/image).
            box_linewidth: Width of the box lines.
            filter_classes: Optionally filter the classes to render.
            max_dist: Maximum distance in meters to still draw a box.
            out_path: Optional path to save the rendered figure to disk.
            render_2d: Whether to render 2d boxes (only works for camera data).

        """
        # Default settings.
        if color_func is None:
            color_func = LyftDatasetExplorer.get_color

        boxes = self.get_boxes(token, filter_classes=filter_classes, max_dist=max_dist)  # In nuScenes lidar frame.

        if sensor_modality == "lidar":
            # Load pointcloud.
            pc = self.get_pointcloud(token, self.root)  # In KITTI lidar frame.
            pc.rotate(self.kitti_to_nu_lidar.rotation_matrix)  # In nuScenes lidar frame.
            # Alternative options:
            # depth = pc.points[1, :]
            # height = pc.points[2, :]
            intensity = pc.points[3, :]

            # Project points to view.
            points = view_points(pc.points[:3, :], view_3d, normalize=False)
            coloring = intensity

            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            if not augment_previous:
                ax.scatter(points[0, :], points[1, :], c=coloring, s=1)
                ax.set_xlim(-axes_limit, axes_limit)
                ax.set_ylim(-axes_limit, axes_limit)

            if with_anns:
                for box in boxes:
                    color = np.array(color_func(box.name)) / 255
                    box.render(ax, view=view_3d, colors=(color, color, "k"), linewidth=box_linewidth)

        elif sensor_modality == "camera":
            im_path = KittiDB.get_filepath(token, "image_2", root=self.root)
            im = Image.open(im_path)

            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(16, 9))

            if not augment_previous:
                ax.imshow(im)
                ax.set_xlim(0, im.size[0])
                ax.set_ylim(im.size[1], 0)

            if with_anns:
                if render_2d:
                    # Use KITTI's 2d boxes.
                    boxes_2d, names = self.get_boxes_2d(token, filter_classes=filter_classes)
                    for box, name in zip(boxes_2d, names):
                        color = np.array(color_func(name)) / 255
                        ax.plot([box[0], box[0]], [box[1], box[3]], color=color, linewidth=box_linewidth)
                        ax.plot([box[2], box[2]], [box[1], box[3]], color=color, linewidth=box_linewidth)
                        ax.plot([box[0], box[2]], [box[1], box[1]], color=color, linewidth=box_linewidth)
                        ax.plot([box[0], box[2]], [box[3], box[3]], color=color, linewidth=box_linewidth)
                else:
                    # Project 3d boxes to 2d.
                    transforms = self.get_transforms(token, self.root)
                    for box in boxes:
                        # Undo the transformations in get_boxes() to get back to the camera frame.
                        box.rotate(self.kitti_to_nu_lidar_inv)  # In KITTI lidar frame.
                        box.rotate(Quaternion(matrix=transforms["velo_to_cam"]["R"]))
                        box.translate(transforms["velo_to_cam"]["T"])  # In KITTI camera frame, un-rectified.
                        box.rotate(Quaternion(matrix=transforms["r0_rect"]))  # In KITTI camera frame, rectified.

                        # Filter boxes outside the image (relevant when visualizing nuScenes data in KITTI format).
                        if not box_in_image(box, transforms["p_left"][:3, :3], im.size, vis_level=BoxVisibility.ANY):
                            continue

                        # Render.
                        color = np.array(color_func(box.name)) / 255
                        box.render(
                            ax,
                            view=transforms["p_left"][:3, :3],
                            normalize=True,
                            colors=(color, color, "k"),
                            linewidth=box_linewidth,
                        )
        else:
            raise ValueError("Unrecognized modality {}.".format(sensor_modality))

        ax.axis("off")
        ax.set_title(token)
        ax.set_aspect("equal")

        # Render to disk.
        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path)
