import math

from pyquaternion import Quaternion

from lyft_dataset_sdk.utils.data_classes import Box


def test_box_rotation():
    q1 = Quaternion(axis=[0, 0, 1])
    q2 = Quaternion(axis=[0, 0, 1], angle=math.pi / 6)

    b = Box(center=(2, 0, 0.5), size=(1, 1, 1), orientation=q1)
    b.rotate_around_box_center(q2)

    b1 = Box(center=(0, 0, 0.5), size=(1, 1, 1), orientation=q2)
    b1.translate((2, 0, 0))

    assert b == b1


def test_rotate_around_origin():
    q1 = Quaternion(axis=[0, 0, 1], angle=math.pi / 5)
    q2 = Quaternion(axis=[0, 0, 1], angle=math.pi / 6)
    b = Box(center=(2, 0, 0.5), size=(1, 1, 1), orientation=q1)
    b.rotate_around_origin(q2)

    assert b == Box(
        center=(2 * math.cos(math.pi / 6), 2 * math.sin(math.pi / 6), 0.5), size=(1, 1, 1), orientation=q1 * q2
    )
