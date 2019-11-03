"""
In https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles

Kaggle expects the submission to look as a csv file with columns

Id	PredictionString

where `Id` corresponds to the sample_token
and `PredictionString` is join of the predictions in the format:

[score center_x center_y center_z width length height yaw class_name] for predictions

and

[center_x center_y center_z width length height yaw class_name] for the ground truth file.

ex:

97ce3ab08ccbc0baae0267cbf8d4da947e1f11ae1dbcb80c3f4408784cd9170c,
1.0 2742.152625996093 673.1631800662494 -18.6561112411676 1.834 4.609 1.648 2.619835541569646 car

This script allow mapping from the kaggle format to the nuscences:

gt = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [974.2811881299899, 1714.6815014457964, -23.689857123368846],
    'size': [1.796, 4.488, 1.664],
    'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
    'name': 'car'
}]

prediction_result = {
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
    'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
    'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
    'name': 'car',
    'score': 0.3077029437237213
}

"""

import argparse
import json

import numpy as np
import pandas as pd
from pyquaternion import Quaternion


def parse_args():
    parser = argparse.ArgumentParser("Convert annotations from Kaggle to Nuscences.")
    arg = parser.add_argument
    arg("-i", "--input_file", type=str, help="Path to the input file.", required=True)
    arg("-o", "--output_file", type=str, help="Path to the output file.", required=True)
    arg("-t", "--type", type=str, help="Predictions or ground truth.", required=True, choices=["pred", "gt"])

    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)

    if args.type == "pred":
        num_parameters_in_bbox = 9
        columns = ["score", "center_x", "center_y", "center_z", "width", "length", "height", "yaw", "name"]
        target_columns = ["score", "sample_token", "translation", "size", "rotation", "name"]
    elif args.type == "gt":
        num_parameters_in_bbox = 8
        columns = ["center_x", "center_y", "center_z", "width", "length", "height", "yaw", "name"]
        target_columns = ["sample_token", "translation", "size", "rotation", "name"]
    else:
        raise NotImplementedError("Only `pred` and `gt` supported for bbox type.")

    temp = []

    for i in df.index:
        bbox_string = df.loc[i, "PredictionString"].strip().split(" ")
        new_shape = (int(len(bbox_string) / num_parameters_in_bbox), num_parameters_in_bbox)

        boxes = np.array(bbox_string).reshape(new_shape)
        dft = pd.DataFrame(boxes, columns=columns)
        dft["sample_token"] = df.loc[i, "Id"]
        temp += [dft]

    joined_df = pd.concat(temp).reset_index(drop=True)

    joined_df["center_x"] = joined_df["center_x"].astype(float)
    joined_df["center_y"] = joined_df["center_y"].astype(float)
    joined_df["center_z"] = joined_df["center_z"].astype(float)
    joined_df["width"] = joined_df["width"].astype(float)
    joined_df["length"] = joined_df["length"].astype(float)
    joined_df["height"] = joined_df["height"].astype(float)
    joined_df["yaw"] = joined_df["yaw"].astype(float)
    if args.type == "pred":
        joined_df["score"] = joined_df["score"].astype(float)

    joined_df["translation"] = joined_df.apply(lambda x: [x["center_x"], x["center_y"], x["center_z"]], 1)
    joined_df["size"] = joined_df.apply(lambda x: [x["width"], x["length"], x["height"]], 1)
    joined_df["rotation"] = joined_df["yaw"].apply(lambda x: Quaternion(axis=[0, 0, 1], angle=x).elements.tolist(), 1)

    joined_df = joined_df[target_columns]

    with open(args.output_file, "w") as f:
        json.dump(joined_df.to_dict("records"), f, indent=4)


if __name__ == "__main__":
    main()
