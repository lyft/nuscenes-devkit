import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pyquaternion import Quaternion

from lyft_dataset_sdk.utils.data_classes import Box

pi = math.pi


def get_inverse_q(q: Quaternion) -> Quaternion:
    return Quaternion(axis=q.axis, angle=-q.angle)


@given(
    size=arrays(
        np.float32,
        3,
        st.floats(min_value=1, max_value=100, exclude_min=True, allow_infinity=False, allow_nan=False, width=16),
    ),
    center=arrays(
        np.float32,
        3,
        st.floats(min_value=-10, max_value=10, exclude_min=True, allow_infinity=False, allow_nan=False, width=32),
    ),
    translate=arrays(
        np.float32,
        3,
        st.floats(min_value=-10, max_value=10, exclude_min=True, allow_infinity=False, allow_nan=False, width=32),
    ),
    angle1=st.floats(min_value=0, max_value=pi, allow_infinity=False, allow_nan=False, width=64),
    angle2=st.floats(min_value=0, max_value=pi, allow_infinity=False, allow_nan=False, width=64),
)
def test_rotate_around_box_center(size, center, translate, angle1, angle2):
    axis = [0, 0, 1]

    q1 = Quaternion(axis=axis, angle=angle1)
    q2 = Quaternion(axis=axis, angle=angle2)

    minus_q2 = Quaternion(axis=axis, angle=-q2.angle)

    original = Box(center=center, size=size, orientation=q1)

    assert original == (original.copy().rotate_around_box_center(q2).rotate_around_box_center(minus_q2))

    assert original == (
        original.copy()
        .rotate_around_box_center(q2)
        .translate(translate)
        .rotate_around_box_center(minus_q2)
        .translate(-translate)
    )


@given(
    size=arrays(
        np.float32,
        3,
        st.floats(min_value=0, max_value=100, exclude_min=True, allow_infinity=False, allow_nan=False, width=16),
    ),
    center=arrays(
        np.float32,
        3,
        st.floats(min_value=-10, max_value=10, exclude_min=True, allow_infinity=False, allow_nan=False, width=32),
    ),
    angle1=st.floats(min_value=0, max_value=pi, allow_infinity=False, allow_nan=False, width=64),
    angle2=st.floats(min_value=0, max_value=pi, allow_infinity=False, allow_nan=False, width=64),
)
def test_rotate_around_origin_xy(size, angle1, angle2, center):
    x, y, z = center

    axis = [0, 0, 1]

    q1 = Quaternion(axis=axis, angle=angle1)
    q2 = Quaternion(axis=axis, angle=angle2)

    minus_q2 = Quaternion(axis=axis, angle=-q2.angle)

    original = Box(center=(x, y, z), size=size, orientation=q1)

    assert original == (original.copy().rotate_around_box_center(q2).rotate_around_box_center(minus_q2))

    cos_angle2 = q2.rotation_matrix[0, 0]
    sin_angle2 = q2.rotation_matrix[1, 0]

    new_center = x * cos_angle2 - y * sin_angle2, x * sin_angle2 + y * cos_angle2, z
    new_orientation = q1 * q2

    target = Box(center=new_center, size=size, orientation=new_orientation)

    assert original.rotate_around_origin(q2) == target
