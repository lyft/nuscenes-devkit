"""The script extracts the data for one scene.
The script expects the data in the format:

maps

train_data
    attribute.json
    calibrated_sensor.json
    category.json
    ego_pose.json
    instance.json
    log.json
    map.json
    sample_annotation.json
    sample_data.json
    sample.json
    scene.json
    sensor.json
    visibility.json
train_images
    <file_name>.jpeg
train_lidar
    <file_name>.bin
train.csv
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser("Extract one scene from the dataset")
    arg = parser.add_argument
    arg(
        "-s",
        "--scene_token",
        type=str,
        help="Scene token for the scene to be extracted. " "If not specified first token is used",
    )
    arg("-d", "--data_path", type=Path, help="Path to the folder with the extracted dataset.", required=True)
    arg("-o", "--output_path", type=Path, help="Path where to save the data.", required=True)
    return parser.parse_args()


def main():
    args = get_args()
    output_path = args.output_path

    output_json_path = output_path / "train_data"
    output_json_path.mkdir(exist_ok=True, parents=True)

    output_image_path = output_path / "train_images"
    output_image_path.mkdir(exist_ok=True, parents=True)

    output_lidar_path = output_path / "train_lidar"
    output_lidar_path.mkdir(exist_ok=True, parents=True)

    for file_name in ["attribute.json", "visibility.json", "category.json", "log.json", "map.json"]:
        shutil.copy(str(args.data_path / "train_data" / file_name), str(output_json_path / file_name))

    (output_path / "train_maps").mkdir(exist_ok=True, parents=True)
    shutil.copy(
        str(args.data_path / "train_maps" / "map_raster_palo_alto.png"),
        str(output_path / "train_maps" / "map_raster_palo_alto.png"),
    )

    with open(args.data_path / "train_data" / "calibrated_sensor.json") as f:
        calibrated_sensor_df = pd.DataFrame(json.load(f))

    with open(args.data_path / "train_data" / "ego_pose.json") as f:
        ego_pose_df = pd.DataFrame(json.load(f))

    with open(args.data_path / "train_data" / "instance.json") as f:
        instance_df = pd.DataFrame(json.load(f))

    with open(args.data_path / "train_data" / "sample_annotation.json") as f:
        sample_annotation_df = pd.DataFrame(json.load(f))

    with open(args.data_path / "train_data" / "sample_data.json") as f:
        sample_data_df = pd.DataFrame(json.load(f))

    with open(args.data_path / "train_data" / "sample.json") as f:
        sample_df = pd.DataFrame(json.load(f))

    with open(args.data_path / "train_data" / "scene.json") as f:
        scene_df = pd.DataFrame(json.load(f))

    with open(args.data_path / "train_data" / "sensor.json") as f:
        sensor_df = pd.DataFrame(json.load(f))

    train_df = pd.read_csv(args.data_path / "train.csv")

    if args.scene_token is None or args.scene_token not in scene_df["token"]:
        scene_token = scene_df["token"].values[0]
    else:
        scene_token = args.scene_token

    scene_df = scene_df[scene_df["token"] == scene_token]
    with open(output_json_path / "scene.json", "w") as f:
        json.dump(scene_df.to_dict(orient="records"), f)

    sample_df = sample_df[sample_df["scene_token"] == scene_token]
    with open(output_json_path / "sample.json", "w") as f:
        json.dump(sample_df.to_dict(orient="records"), f)

    valid_sample_tokens = set(sample_df["token"].values)

    sample_data_df = sample_data_df[sample_data_df["sample_token"].isin(valid_sample_tokens)]

    sample_data_df["filename"] = (
        sample_data_df["filename"].str.replace("lidar/", "train_lidar/").str.replace("images/", "train_images/")
    )

    with open(output_json_path / "sample_data.json", "w") as f:
        json.dump(sample_data_df.to_dict(orient="records"), f)

    valid_calibrated_sensor_tokens = set(sample_data_df["calibrated_sensor_token"])
    valid_ego_pose_tokens = set(sample_data_df["ego_pose_token"])

    lidar_file_names = sample_data_df.loc[sample_data_df["fileformat"] == "bin", "filename"].values
    for file_name in lidar_file_names:
        shutil.copy(
            str(args.data_path / "train_lidar" / Path(file_name).name), str(output_lidar_path / Path(file_name).name)
        )

    image_file_names = sample_data_df.loc[sample_data_df["fileformat"] == "jpeg", "filename"].values
    for file_name in image_file_names:
        shutil.copy(
            str(args.data_path / "train_images" / Path(file_name).name), str(output_image_path / Path(file_name).name)
        )

    train_df = train_df[train_df["Id"].isin(valid_sample_tokens)]
    train_df.to_csv(output_path / "train.csv", index=False)

    calibrated_sensor_df = calibrated_sensor_df[calibrated_sensor_df["token"].isin(valid_calibrated_sensor_tokens)]
    with open(output_json_path / "calibrated_sensor.json", "w") as f:
        json.dump(calibrated_sensor_df.to_dict(orient="records"), f)

    valid_sensor_tokens = set(calibrated_sensor_df["sensor_token"])

    sensor_df = sensor_df[sensor_df["token"].isin(valid_sensor_tokens)]
    with open(output_json_path / "sensor.json", "w") as f:
        json.dump(sensor_df.to_dict(orient="records"), f)

    ego_pose_df = ego_pose_df[ego_pose_df["token"].isin(valid_ego_pose_tokens)]
    with open(output_json_path / "ego_pose.json", "w") as f:
        json.dump(ego_pose_df.to_dict(orient="records"), f)

    sample_annotation_df = sample_annotation_df[sample_annotation_df["sample_token"].isin(valid_sample_tokens)]
    with open(output_json_path / "sample_annotation.json", "w") as f:
        json.dump(sample_annotation_df.to_dict(orient="records"), f)

    valid_instance_tokens = set(sample_annotation_df["instance_token"])

    instance_df = instance_df[instance_df["token"].isin(valid_instance_tokens)]
    with open(output_json_path / "instance.json", "w") as f:
        json.dump(instance_df.to_dict(orient="records"), f)


if __name__ == "__main__":
    main()
