import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from torchvision import transforms
from mahe_mobility.tasks.task1_lidar_to_occupancy import load_lidar_ego_frame, lidar_to_occupancy
from mahe_mobility.geometry.lss_core import DepthConfig, CameraConfig

# --- Shared Transforms & Noise ---
class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.03):
        self.mean, self.std = mean, std
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

_IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 480)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    AddGaussianNoise(std=0.02),
    _IMAGENET_NORMALIZE,
])

_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 480)),
    transforms.ToTensor(),
    _IMAGENET_NORMALIZE,
])

class NuScenesFrontCameraDataset(Dataset):
    def __init__(self, dataroot="./data/nuscenes", version="v1.0-mini", train: bool = True):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.samples = self.nusc.sample
        self.transform = _TRAIN_TRANSFORM if train else _VAL_TRANSFORM
        self.depth_cfg = DepthConfig() # Uses default d_min, d_max, d_steps
        self.cam_cfg = CameraConfig(image_h=224, image_w=480)
        
        # Accuracy Push: BEV-space augmentation settings
        self.is_train = train
        self.aug_rot_range = [-15, 15]  # degrees
        self.aug_scale_range = [0.95, 1.05]
        
        mode = "training" if train else "validation"
        print(f"✅ NuScenes Dataset Ready: Found {len(self.samples)} samples [{mode} mode]")

    def __len__(self):
        return len(self.samples)

    def get_depth_gt(self, pts_ego, intrinsic, translation, rotation):
        """Projects LiDAR points to Image and creates binned depth GT."""
        # 1. Transform Ego -> Cam
        from pyquaternion import Quaternion
        quat = Quaternion(rotation.numpy())
        cam2ego = np.eye(4)
        cam2ego[:3, :3] = quat.rotation_matrix
        cam2ego[:3, 3] = translation.numpy()
        ego2cam = np.linalg.inv(cam2ego)
        
        # pts_ego is (N, 3). Convert to (3, N)
        pts_cam = np.dot(ego2cam[:3, :3], pts_ego.T) + ego2cam[:3, 3:4]
        
        # 2. Project to Image
        pts_img = view_points(pts_cam, intrinsic.numpy(), normalize=True)
        
        # 3. Filter points
        mask = np.ones(pts_img.shape[1], dtype=bool)
        mask &= (pts_cam[2, :] > self.depth_cfg.d_min)
        mask &= (pts_cam[2, :] < self.depth_cfg.d_max)
        mask &= (pts_img[0, :] >= 0)
        mask &= (pts_img[0, :] < self.cam_cfg.image_w)
        mask &= (pts_img[1, :] >= 0)
        mask &= (pts_img[1, :] < self.cam_cfg.image_h)
        
        depths = pts_cam[2, mask]
        coords = pts_img[:2, mask].astype(int)
        
        # 4. Create Depth Map
        depth_gt = torch.zeros((self.cam_cfg.image_h, self.cam_cfg.image_w))
        
        # Depth d maps to bin = floor((d - d_min) / d_step)
        # d_step = (d_max - d_min) / (d_steps - 1)
        d_step = (self.depth_cfg.d_max - self.depth_cfg.d_min) / (self.depth_cfg.d_steps - 1)
        
        bin_indices = np.floor((depths - self.depth_cfg.d_min) / d_step).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.depth_cfg.d_steps - 1)
        
        depth_gt[coords[1], coords[0]] = torch.tensor(bin_indices).float() + 1 # 0 is 'no point'
        
        return depth_gt

    def __getitem__(self, idx):
        my_sample = self.samples[idx]
        cam_data = self.nusc.get("sample_data", my_sample["data"]["CAM_FRONT"])

        # Load Image
        img_path = self.nusc.get_sample_data_path(my_sample["data"]["CAM_FRONT"])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        # Load Geometry (Intrinsics + Extrinsics)
        calibrated = self.nusc.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )
        intrinsic = torch.tensor(calibrated["camera_intrinsic"], dtype=torch.float32)
        translation = torch.tensor(calibrated["translation"], dtype=torch.float32)
        rotation = torch.tensor(calibrated["rotation"], dtype=torch.float32)

        sample_token = my_sample["token"]
        pts_ego = load_lidar_ego_frame(self.nusc, sample_token)
        
        # ── Accuracy Push: BEV-Space Augmentation ───────────────────
        if self.is_train:
            # 1. Sample random rotation and scale
            rot_deg = np.random.uniform(*self.aug_rot_range)
            scale = np.random.uniform(*self.aug_scale_range)
            
            rot_rad = np.radians(rot_deg)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            
            # Pure rotation matrix (for extrinsics sync)
            R_aug_pure = np.array([
                [cos_r, -sin_r, 0],
                [sin_r,  cos_r, 0],
                [0,      0,     1]
            ])
            
            # Scaled rotation matrix (for LiDAR/BEV points)
            R_aug_scaled = R_aug_pure * scale
            
            # 2. Transform LiDAR points (Ego frame)
            pts_ego = (R_aug_scaled @ pts_ego.T).T
            
            # 3. SYNC: Update Extrinsics for the LSS model
            # Must be PURE rotation for Quaternion constructor
            from pyquaternion import Quaternion
            q_ext = Quaternion(rotation.numpy())
            R_ext_mat = q_ext.rotation_matrix
            R_new_mat = R_ext_mat @ R_aug_pure.T
            rotation = torch.tensor(Quaternion(matrix=R_new_mat).elements).float()
            
            # 4. SYNC: Update Intrinsics (scale focal lengths to match BEV zoom)
            intrinsic[0, 0] *= scale
            intrinsic[1, 1] *= scale
        
        # Generate Depth GT (Uses the potentially augmented pts_ego)
        gt_depth = self.get_depth_gt(pts_ego, intrinsic, translation, rotation)
        
        # Generate BEV Occupancy (Uses the potentially augmented pts_ego)
        gt_occupancy = lidar_to_occupancy(pts_ego)
        gt_occupancy_tensor = torch.tensor(gt_occupancy, dtype=torch.float32).unsqueeze(0)

        return image_tensor, intrinsic, translation, rotation, gt_occupancy_tensor, gt_depth