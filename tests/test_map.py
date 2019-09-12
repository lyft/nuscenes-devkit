import math
from lyft_dataset_sdk.eval.detection.mAP_evaluation import recall_precision, group_by_key, Box3D
import json
import numpy as np
import pytest


@pytest.mark.parametrize(
    "rotation",
    [
        [1, 0, 0, 0],  # angle = 0
        [0, 0, 0, 1],  # 180 rotation around z axis
        [math.sqrt(0.5), 0, 0, math.sqrt(0.5)],  # 90 rotation around z axis
    ],
)
def test_fround_area(rotation):
    translation = [0, 0, 1.5]
    size = 2, 4, translation[2] * 2  # width, length, height

    sample_token = ""
    name = "car"

    b = Box3D(translation=translation, rotation=rotation, size=size, sample_token=sample_token, name=name)

    assert b.volume == np.prod(size)


@pytest.mark.parametrize(
    "rotation",
    [
        [1, 0, 0, 0],  # angle = 0
        [0, 0, 0, 1],  # 180 rotation around z axis
        [math.sqrt(0.5), 0, 0, math.sqrt(0.5)],  # 90 rotation around z axis
    ],
)
def test_ground_area(rotation):
    translation = [0, 0, 1.5]
    size = 2, 4, translation[2] * 2  # width, length, height

    sample_token = ""
    name = "car"

    b = Box3D(translation=translation, rotation=rotation, size=size, sample_token=sample_token, name=name)

    assert b.volume == np.prod(size)

    assert np.isclose(b.ground_bbox_coords.area, b.length * b.width, rtol=1e-05, atol=1e-08, equal_nan=False)


@pytest.mark.parametrize(
    ["target_box", "intersection"],
    [
        (Box3D(translation=[0, 0, 1.5], size=[2, 4, 3], rotation=[1, 0, 0, 0], name="car", sample_token=""), 24),
        (Box3D(translation=[4, 0, 1.5], size=[2, 4, 3], rotation=[1, 0, 0, 0], name="car", sample_token=""), 0),
        (Box3D(translation=[0, 0, 1.5], size=[2, 2, 3], rotation=[1, 0, 0, 0], name="car", sample_token=""), 12),
        (Box3D(translation=[0, 0, 1.5], size=[2, 4, 3], rotation=[0, 0, 0, 1], name="car", sample_token=""), 24),
        (
            Box3D(
                translation=[0, 0, 1.5],
                size=[2, 4, 3],
                rotation=[math.sqrt(0.5), 0, 0, math.sqrt(0.5)],
                name="car",
                sample_token="",
            ),
            12,
        ),
    ],
)
def test_intersection(target_box, intersection):
    original_box = Box3D(translation=[0, 0, 1.5], size=[2, 4, 3], rotation=[1, 0, 0, 0], name="car", sample_token="")
    assert np.isclose(original_box.get_intersection(target_box), intersection, rtol=1e-05, atol=1e-08, equal_nan=False)


def test_self():
    original_box = Box3D(
        sample_token="",
        translation=[2680.2830359779, 698.1969292853, -18.0477669237],
        size=[2.064, 5.488, 2.053],
        rotation=[0.2654919368, 0, 0, 0.9641130802],
        name="car",
    )

    assert original_box.volume == np.prod([2.064, 5.488, 2.053])
    assert np.isclose(original_box.ground_bbox_coords.area, 2.064 * 5.488)

    assert np.isclose(original_box.get_area_intersection(original_box), 2.064 * 5.488)
    assert np.isclose(original_box.get_height_intersection(original_box), 2.053)
    assert np.isclose(original_box.get_intersection(original_box), original_box.volume)
    assert np.isclose(original_box.get_iou(original_box), 1)


@pytest.mark.parametrize(
    ["original_box", "target_box", "intersection", "height_intersection", "area_intersection"],
    [
        (
            Box3D(
                sample_token="",
                translation=[2680.2830359779, 698.1969292853, -18.0477669237],
                size=[2.064, 5.488, 2.053],
                rotation=[0.2654919368, 0, 0, 0.9641130802],
                name="car",
            ),
            Box3D(
                sample_token="",
                translation=[2680.2830359779, 698.1969292853, -18.0477669237],
                size=[2.064, 5.488, 2.053],
                rotation=[0.2654919368, 0, 0, 0.9641130802],
                name="car",
            ),
            23.254807296,
            2.053,
            2.064 * 5.488,
        )
    ],
)
def test_intersection_couples(original_box, target_box, intersection, height_intersection, area_intersection):
    assert np.isclose(
        original_box.get_height_intersection(target_box), height_intersection, rtol=1e-05, atol=1e-08, equal_nan=False
    )

    assert np.isclose(
        original_box.get_area_intersection(target_box), area_intersection, rtol=1e-05, atol=1e-08, equal_nan=False
    )


@pytest.mark.parametrize(
    ["target_box", "iou"],
    [
        (Box3D(translation=[0, 0, 1.5], size=[2, 4, 3], rotation=[1, 0, 0, 0], name="car", sample_token=""), 1),
        (Box3D(translation=[4, 0, 1.5], size=[2, 4, 3], rotation=[1, 0, 0, 0], name="car", sample_token=""), 0),
        # same but twice smaller
        (Box3D(translation=[0, 0, 1.5], size=[2, 2, 3], rotation=[1, 0, 0, 0], name="car", sample_token=""), 0.5),
        # rotate by 180
        (Box3D(translation=[0, 0, 1.5], size=[2, 4, 3], rotation=[0, 0, 0, 1], name="car", sample_token=""), 1),
        # rotated by 90
        (
            Box3D(
                translation=[0, 0, 1.5],
                size=[2, 4, 3],
                rotation=[math.sqrt(0.5), 0, 0, math.sqrt(0.5)],
                name="car",
                sample_token="",
            ),
            1 / 3,
        ),
    ],
)
def test_iou(target_box, iou):
    original_box = Box3D(translation=[0, 0, 1.5], size=[2, 4, 3], rotation=[1, 0, 0, 0], name="car", sample_token="")
    assert np.isclose(original_box.get_iou(target_box), iou, rtol=1e-05, atol=1e-08, equal_nan=False)


@pytest.mark.parametrize("iou_threshold", np.arange(0, 1, 0.05))
def test_itself(iou_threshold):
    """Same boxes, with all scores being the same should give 1 as mAP"""
    with open("tests/test_jsons/samples_merged.json") as f:
        list_boxes = json.load(f)

    class_name = "car"

    for item in list_boxes:
        item["score"] = 1

    gt_by_class_name = group_by_key(list_boxes, "name")
    pred_by_class_name = group_by_key(list_boxes, "name")

    recalls, precisions, average_precision = recall_precision(
        gt_by_class_name[class_name], pred_by_class_name[class_name], iou_threshold
    )

    assert average_precision == 1


# json files for test

# Ground truth file - Not all classes present
gt_file_without_all_classes = "tests/test_jsons/true_gt.json"

# Prediction same as above ground truth
true_res_without_all_classes = "tests/test_jsons/test_file_for_true_result.json"

# Ground truth file - All classes present
gt_file_with_all_classes = "tests/test_jsons/all_class_gt.json"

# Prediction same as above ground truth
true_res_with_all_classes = "tests/test_jsons/test_file_with_all_classes.json"

# gt_wil_all_classes, where all names replaced by "car"
res_file_with_wrong_name = "tests/test_jsons/test_file_with_one_class.json"

# Prediction file is missing one prediction
res_file_with_false_neg = "tests/test_jsons/test_file_for_false_neg.json"

# Prediction has an extra prediction of class present in gt
res_file_with_fp_gt = "tests/test_jsons/test_file_for_false_pos_gt.json"

# Prediction has an extra prediction of class not present in gt
res_file_with_fp_random = "tests/test_jsons/test_file_for_false_pos_random.json"


def calc_map(gt_file, pred_file, iou_threshold=0.5):
    with open(pred_file) as f:
        predictions = json.load(f)

    with open(gt_file) as f:
        gt = json.load(f)

    gt_by_class_name = group_by_key(gt, "name")
    pred_by_class_name = group_by_key(predictions, "name")

    class_names = sorted(gt_by_class_name.keys())

    average_precisions = np.zeros(len(class_names))

    for class_id, class_name in enumerate(class_names):
        if class_name in pred_by_class_name:
            _, _, avg_prec = recall_precision(
                gt_by_class_name[class_name], pred_by_class_name[class_name], iou_threshold
            )
            average_precisions[class_id] = avg_prec

    mAP = np.mean(average_precisions)

    return mAP


@pytest.mark.parametrize(
    "iou", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
)
def test_ground_truth(iou):
    """
        Test for ground truth
    """
    gt_file = gt_file_without_all_classes
    pred_file = true_res_without_all_classes
    mAP = calc_map(gt_file, pred_file, iou)

    # Predicition and ground truth contains same samples
    assert mAP == 1


@pytest.mark.parametrize(
    "iou", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
)
def test_ground_truth_with_all_classes(iou):
    """
        Test for ground truth with all classes present
    """
    gt_file = gt_file_with_all_classes
    pred_file = true_res_with_all_classes
    mAP = calc_map(gt_file, pred_file, iou)

    # Predicition and ground truth contains same samples
    assert mAP == 1


# def test_ground_truth_with_one_class_in_pred():
#     """
#         Test for ground truth with all classes present.
#
#         Prediction and ground truth contains same bounding boxes.
#         But all are predicted as car.
#
#     """
#     gt_file = gt_file_with_all_classes
#     pred_file = res_file_with_wrong_name
#     mAP = calc_map(gt_file, pred_file)
#
#     # AP for car = 1/9
#     # AP for all other classes = 0
#     assert mAP == ((0 * 8) + 1 / 9) / 9


# def test_false_positive_gt():
#     """
#         Test for false positive when the extra prediction is a class
#         in ground truth
#     """
#     gt_file = gt_file_without_all_classes
#     pred_file = res_file_with_fp_gt
#     mAP = calc_map(gt_file, pred_file)
#
#     # 5 classes are not present in ground truth and prediction - AP = 1
#     # There are 3 classes predicted correctly - AP = 1
#     # There are 1 car in ground truth and 2 cars in prediction - AP = 1 / 2
#     assert mAP == ((1 * 8) + 0.5) / 9


# def test_false_positive_random():
#     """
#         Test for false positive when the extra prediction is a class
#         not in ground truth
#     """
#     gt_file = gt_file_without_all_classes
#     pred_file = res_file_with_fp_random
#     mAP = calc_map(gt_file, pred_file)
#
#     # 4 classes are not present in ground truth and prediction - AP = 1
#     # There are 4 classes in ground predicted correctly - AP = 1
#     # There is no sample for truck in ground truth,
#     # but 1 truck in prediction - AP = 0
#     assert mAP == ((1 * 8) + 0) / 9


# def test_false_negative():
#     """
#         Test for the case when there is the prediction is missing
#         a sample
#     """
#     gt_file = gt_file_without_all_classes
#     pred_file = res_file_with_false_neg
#     mAP = calc_map(gt_file, pred_file)
#
#     # 5 classes are not present in ground truth and prediction - AP = 1
#     # There are 3 classes in ground truth predicted correctly - AP = 1
#     # There is 1 class in ground truth missing in predicition - AP = 0
#     assert mAP == ((1 * 8) + 0) / 9


def get_ground_truth_and_pred_box():
    gt_file = gt_file_with_all_classes
    pred_file = true_res_with_all_classes

    with open(gt_file, "r") as ground_truth:
        data = json.load(ground_truth)

    ground_truth_box = Box3D(
        sample_token=data[0]["sample_token"],
        translation=data[0]["translation"],
        size=data[0]["size"],
        rotation=data[0]["rotation"],
        name=data[0]["name"],
    )

    with open(pred_file, "r") as prediction_file:
        data = json.load(prediction_file)

    prediction_box = Box3D(
        sample_token=data[0]["sample_token"],
        translation=data[0]["translation"],
        size=data[0]["size"],
        rotation=data[0]["rotation"],
        name=data[0]["name"],
    )

    return ground_truth_box, prediction_box


def modify_prediction_and_get_box(translation=[0, 0, 0], size=[1, 1, 1], rotation=[0, 0, 0, 1]):
    ground_truth_box = Box3D(
        sample_token="a3b278456a7ee38322388eda31378d0c91a48645fba18b8",
        translation=[0, 0, 0],
        size=[1, 1, 1],
        rotation=[0, 0, 0, 1],
        name="animal",
    )

    prediction_box = Box3D(
        sample_token="a3b278456a7ee38322388eda31378d0c91a48645fba18b8",
        translation=translation,
        size=size,
        rotation=rotation,
        name="animal",
    )

    return ground_truth_box, prediction_box


def test_translation_change_x():
    """
        Move the center_x of prediction and test for iou
    """
    ground_truth_box, prediction_box = modify_prediction_and_get_box(translation=[0.5, 0, 0])

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 0.3333333, 1e-7)


def test_translation_change_y():
    """
        Move the center_y of prediction and test for iou
    """
    ground_truth_box, prediction_box = modify_prediction_and_get_box(translation=[0, 0.5, 0])

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 0.3333333, 1e-7)


def test_translation_change_z():
    """
        Move the center_z of prediction and test for iou
    """
    ground_truth_box, prediction_box = modify_prediction_and_get_box(translation=[0, 0, 0.5])

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 0.3333333, 1e-7)


def test_width_change():
    """
        Change width and test for iou
    """
    ground_truth_box, prediction_box = modify_prediction_and_get_box(size=[2, 1, 1])

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 0.5, 1e-7)


def test_length_change():
    """
        Change length and test for iou
    """
    ground_truth_box, prediction_box = modify_prediction_and_get_box(size=[1, 2, 1])

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 0.5, 1e-7)


def test_height_change():
    """
        Change height and test for iou
    """
    ground_truth_box, prediction_box = modify_prediction_and_get_box(size=[1, 1, 2])

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 0.5, 1e-7)


def test_touch_edge_bounding_boxes():
    """
        test iou for bounding boxes meeting at edge
    """
    ground_truth_box, prediction_box = modify_prediction_and_get_box(translation=[1, 1, 0])

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 0, 1e-7)


def test_touch_side_bounding_boxes():
    """
        test iou for bounding boxes meeting at side
    """
    ground_truth_box, prediction_box = modify_prediction_and_get_box(translation=[0, 1, 0])

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 0, 1e-7)


def test_touch_corner_bounding_boxes():
    """
        test iou for bounding boxes meeting at corner
    """
    ground_truth_box, prediction_box = modify_prediction_and_get_box(translation=[1, 1, 1])

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 0, 1e-7)


def test_area_intersection():
    """
        Test Area intersection with itself
    """
    ground_truth_box, prediction_box = get_ground_truth_and_pred_box()

    ground_truth_base = ground_truth_box.get_ground_bbox_coords()

    assert np.isclose(ground_truth_box.get_area_intersection(prediction_box), ground_truth_base.area, 1e-7)


def test_height_intersection():
    """
        Test height intersection with itself
    """
    ground_truth_box, prediction_box = get_ground_truth_and_pred_box()

    assert np.isclose(ground_truth_box.get_height_intersection(prediction_box), ground_truth_box.size[2], 1e-7)


def test_selfintersection():
    """
        Test intersection with itself
    """
    ground_truth_box, prediction_box = get_ground_truth_and_pred_box()

    ground_truth_base = ground_truth_box.get_ground_bbox_coords()

    area_calculated = ground_truth_base.area * ground_truth_box.size[2]

    assert np.isclose(ground_truth_box.get_intersection(prediction_box), area_calculated, 1e-7)


def test_selfiou():
    """
        Test iou with itself
    """
    ground_truth_box, prediction_box = get_ground_truth_and_pred_box()

    assert np.isclose(ground_truth_box.get_iou(prediction_box), 1, 1e-7)
