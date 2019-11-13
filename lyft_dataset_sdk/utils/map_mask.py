# Lyft Dataset SDK
# Code written by Qiang Xu and Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]
# Modified by Vladimir Iglovikov 2019.

from typing import Tuple, Any

import cv2
import numpy as np
from PIL import Image
from cachetools import cached, LRUCache
from pathlib import Path

# Set the maximum loadable image size.
Image.MAX_IMAGE_PIXELS = 400000 * 400000


class MapMask:
    def __init__(self, img_file: (Path, str), resolution: float = 0.1):
        """Init a map mask object that contains the semantic prior (drivable surface and sidewalks) mask.

        Args:
            img_file: File path to map png file.
            resolution: Map resolution in meters.
        """

        self.img_file = Path(img_file)

        if not self.img_file.exists():
            raise FileNotFoundError(f"map mask {img_file} does not exist")

        if resolution < 0.1:
            raise ValueError("Only supports down to 0.1 meter resolution.")

        self.resolution = resolution
        self.background = [255, 255, 255]

    @cached(cache=LRUCache(maxsize=3))
    def mask(self) -> np.ndarray:
        """Returns the map mask."""
        return self._base_mask

    @property
    def transform_matrix(self) -> np.ndarray:
        """Generate transform matrix for this map mask.

        Returns: <np.array: 4, 4>. The transformation matrix.

        """
        return np.array(
            [
                [1.0 / self.resolution, 0, 0, 0],
                [0, -1.0 / self.resolution, 0, self._base_mask.shape[0]],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def is_on_mask(self, x: Any, y: Any) -> np.array:
        """Determine whether the given coordinates are on the (optionally dilated) map mask.

        Args:
            x: Global x coordinates. Can be a scalar, list or a numpy array of x coordinates.
            y: Global y coordinates. Can be a scalar, list or a numpy array of x coordinates.

        Returns: <np.bool: x.shape>. Whether the points are on the mask.

        """
        px, py = self.to_pixel_coords(x, y)

        on_mask = np.ones(px.size, dtype=np.bool)
        this_mask = self.mask()

        on_mask[px < 0] = False
        on_mask[px >= this_mask.shape[1]] = False
        on_mask[py < 0] = False
        on_mask[py >= this_mask.shape[0]] = False

        on_mask[on_mask] = np.all(this_mask[py[on_mask], px[on_mask]] != self.background, axis=-1)

        return on_mask

    def to_pixel_coords(self, x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Maps x, y location in global map coordinates to the map image coordinates.

        Args:
            x: Global x coordinates. Can be a scalar, list or a numpy array of x coordinates.
            y: Global y coordinates. Can be a scalar, list or a numpy array of x coordinates.

        Returns: (px <np.uint8: x.shape>, py <np.uint8: y.shape>). Pixel coordinates in map.

        """
        x = np.array(x)
        y = np.array(y)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if x.shape != y.shape:
            raise ValueError("x.shape != y.shape")

        if not x.ndim == y.ndim == 1:
            raise ValueError("x.ndim and y.ndim should be equal to 1")

        pts = np.stack([x, y, np.zeros(x.shape), np.ones(x.shape)])
        pixel_coords = np.round(np.dot(self.transform_matrix, pts)).astype(np.int32)

        return pixel_coords[0, :], pixel_coords[1, :]

    @property
    @cached(cache=LRUCache(maxsize=1))
    def _base_mask(self) -> np.ndarray:
        """Returns the original binary mask stored in map png file.

        Returns: <np.int8: image.height, image.width>. The binary mask.

        """

        # Pillow allows us to specify the maximum image size above, whereas this is more difficult in OpenCV.
        img = Image.open(self.img_file)

        # Resize map mask to desired resolution.
        native_resolution = 0.1
        size_x = int(img.size[0] / self.resolution * native_resolution)
        size_y = int(img.size[1] / self.resolution * native_resolution)
        img = img.resize((size_x, size_y), resample=Image.NEAREST)

        # Convert to numpy.
        raw_mask = np.array(img)
        return raw_mask
