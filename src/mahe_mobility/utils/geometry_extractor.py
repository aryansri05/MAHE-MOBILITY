from nuscenes.nuscenes import NuScenes
import numpy as np


def extract_camera_geometry():
    # 1. Initialize nuScenes (pointing to the mini dataset you downloaded)
    # Adjust the dataroot path if you saved it somewhere else
    print("Loading nuScenes database...")
    nusc = NuScenes(version="v1.0-mini", dataroot="./data/sets/nuscenes", verbose=False)

    # 2. Grab the first sample (a snapshot in time)
    my_sample = nusc.sample[0]

    # 3. Get the data dictionary for the Front Camera
    cam_front_data = nusc.get("sample_data", my_sample["data"]["CAM_FRONT"])

    # 4. Access the specific sensor calibration data
    calibrated_sensor = nusc.get(
        "calibrated_sensor", cam_front_data["calibrated_sensor_token"]
    )

    # --- EXTRACT THE MATRICES ---

    # Intrinsic Matrix (3x3 array)
    intrinsic_matrix = np.array(calibrated_sensor["camera_intrinsic"])

    # Extrinsic Translation (x, y, z in meters from the ego-vehicle center)
    translation = np.array(calibrated_sensor["translation"])

    # Extrinsic Rotation (Quaternion format: w, x, y, z)
    rotation_quaternion = np.array(calibrated_sensor["rotation"])

    print("\n✅ --- Camera Geometry Successfully Extracted --- ✅")
    print(f"\nIntrinsic Matrix (3x3):\n{intrinsic_matrix}")
    print(f"\nCamera Translation (x,y,z in meters):\n{translation}")
    print(f"\nCamera Rotation (Quaternion):\n{rotation_quaternion}")

    return intrinsic_matrix, translation, rotation_quaternion


if __name__ == "__main__":
    extract_camera_geometry()
