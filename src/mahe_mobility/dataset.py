import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from nuscenes.nuscenes import NuScenes
from torchvision import transforms
from mahe_mobility.tasks.task1_lidar_to_occupancy import load_lidar_ego_frame, lidar_to_occupancy


# ── Custom transform: Gaussian noise injected after ToTensor ─────────────────
class AddGaussianNoise:
    """
    Adds random Gaussian noise to a tensor image.
    Simulates sensor noise, low-light grain, and camera artifacts.
    Applied after ToTensor so input is a float tensor in [0, 1].
    """
    def __init__(self, mean: float = 0.0, std: float = 0.03):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"


# ── Normalization (shared between train and val) ──────────────────────────────
_IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

# ── Training transform — aggressive augmentation for robustness ───────────────
_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 480)),

    # --- Lighting & Colour ---
    transforms.ColorJitter(
        brightness=0.5,   # handles day/night extremes
        contrast=0.5,     # handles fog, overcast, glare
        saturation=0.4,   # handles colour shifts
        hue=0.1,          # slight hue shift
    ),
    transforms.RandomGrayscale(p=0.1),  # forces model to work without colour (e.g. night/IR)
    transforms.RandomAutocontrast(p=0.3),  # simulates camera auto-exposure adjustments

    # --- Blur & Sharpness (simulates focus issues / rain on lens) ---
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

    transforms.ToTensor(),

    # --- Noise (simulates sensor/thermal/dark-current noise) ---
    AddGaussianNoise(mean=0.0, std=0.03),

    _IMAGENET_NORMALIZE,
])

# ── Validation transform — no augmentation, clean baseline ───────────────────
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

        mode = "training" if train else "validation"
        print(f"✅ NuScenes Dataset Ready: Found {len(self.samples)} samples [{mode} mode]")

    def __len__(self):
        return len(self.samples)

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
        gt_occupancy = lidar_to_occupancy(pts_ego)
        gt_occupancy_tensor = torch.tensor(gt_occupancy, dtype=torch.float32).unsqueeze(0)

        return image_tensor, intrinsic, translation, rotation, gt_occupancy_tensor