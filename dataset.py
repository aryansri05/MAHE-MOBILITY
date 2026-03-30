import torch
from torch.utils.data import Dataset
from PIL import Image
from nuscenes.nuscenes import NuScenes
from torchvision import transforms
from task1_lidar_to_occupancy import load_lidar_ego_frame, lidar_to_occupancy

class NuScenesFrontCameraDataset(Dataset):
    def __init__(self, dataroot='./data/nuscenes', version='v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.samples = self.nusc.sample[:15]
        self.transform = transforms.Compose([
            transforms.Resize((224, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        my_sample = self.samples[idx]
        cam_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        
        # Load Image
        img_path = self.nusc.get_sample_data_path(my_sample['data']['CAM_FRONT'])
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Load Geometry (Intrinsics + Extrinsics)
        calibrated = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsic = torch.tensor(calibrated['camera_intrinsic'], dtype=torch.float32)
        translation = torch.tensor(calibrated['translation'], dtype=torch.float32)
        rotation = torch.tensor(calibrated['rotation'], dtype=torch.float32)
        
        sample_token = my_sample['token']
        pts_ego = load_lidar_ego_frame(self.nusc, sample_token)
        gt_occupancy = lidar_to_occupancy(pts_ego)
        gt_occupancy_tensor = torch.tensor(gt_occupancy, dtype=torch.float32).unsqueeze(0)
        
        return image_tensor, intrinsic, translation, rotation, gt_occupancy_tensor