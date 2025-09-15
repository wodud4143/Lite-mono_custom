from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import kornia.augmentation as K
import kornia.geometry.transform as KG
import torch.nn.functional as F



def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def collate_fn(batch):
    color_raw_batch = {}
    frame_indices = batch[0]['color_raw'].keys()
    
    for frame_idx in frame_indices:
        color_raw_batch[frame_idx] = torch.stack([
            item['color_raw'][frame_idx] for item in batch
        ])
        
    collated = {
        'color_raw': color_raw_batch,
        'K_raw': torch.stack([item['K_raw'] for item in batch]),
        'do_color_aug': [item['do_color_aug'] for item in batch],
        'do_flip': [item['do_flip'] for item in batch],
        'folder': [item['folder'] for item in batch],
        'frame_index': [item['frame_index'] for item in batch],
        'side': [item['side'] for item in batch]
    }
    
    if batch[0]['depth_raw'] is not None:
        collated['depth_raw'] = torch.stack([item['depth_raw'] for item in batch])
    
    if batch[0]['stereo_T'] is not None:
        collated['stereo_T'] = torch.stack([item['stereo_T'] for item in batch])
    
    return collated


# region - Kornia Aug
class AugmentationProcessor(torch.nn.Module):
    def __init__(self, height, width, num_scales, is_train=False):
        super().__init__()
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.is_train = is_train
        
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2) 
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        
        if is_train:
            self.color_jitter = K.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast, 
                saturation=self.saturation,
                hue=self.hue,
                p=1.0
            )
        
        self.horizontal_flip = K.RandomHorizontalFlip(p=1.0)
        
        self.resizers = torch.nn.ModuleDict()
        for i in range(num_scales):
            s = 2 ** i
            scale_height = height // s
            scale_width = width // s
            self.resizers[str(i)] = KG.Resize(
                (scale_height, scale_width), 
                interpolation='bicubic',
                antialias=True
            )
    
    
    def forward(self, inputs_batch, metadata_batch):
        batch_size = len(metadata_batch['do_color_aug'])
        device = next(iter(inputs_batch['color_raw'].values())).device
        
        processed_inputs = {}
        for frame_idx, images in inputs_batch['color_raw'].items():
            for b_idx in range(batch_size):
                if metadata_batch['do_flip'][b_idx]:
                    images[b_idx] = torch.flip(images[b_idx], dims=[2])
            
            for scale in range(self.num_scales):
                resized_images = self.resizers[str(scale)](images)
                processed_inputs[("color", frame_idx, scale)] = resized_images
                
                if self.is_train:
                    aug_images = resized_images.clone()
                    aug_mask = torch.tensor(metadata_batch['do_color_aug'], 
                                          device=device, dtype=torch.bool)
                    
                    if aug_mask.any():
                        aug_indices = torch.where(aug_mask)[0]
                        aug_subset = aug_images[aug_indices]
                        aug_subset = self.color_jitter(aug_subset)
                        aug_images[aug_indices] = aug_subset
                    
                    processed_inputs[("color_aug", frame_idx, scale)] = aug_images
                else:
                    processed_inputs[("color_aug", frame_idx, scale)] = resized_images
        
        return processed_inputs
    

# region - GPUProcessor
class GPUDataProcessor:
    def __init__(self, height, width, num_scales, is_train=False, device='cuda'):
        self.device = device
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.is_train = is_train
        
        self.preprocessor = AugmentationProcessor(height, width, num_scales, is_train).to(device)
        
        
    def process_batch(self, inputs_batch):
        gpu_batch = self.move_to_gpu(inputs_batch)
        
        metadata = {
            'do_color_aug': gpu_batch['do_color_aug'],
            'do_flip': gpu_batch['do_flip']
        }
        
        processed_inputs = self.preprocessor(gpu_batch, metadata)
        self.process_camera_params(gpu_batch, processed_inputs)
        
        if 'depth_raw' in gpu_batch and gpu_batch['depth_raw'] is not None:
            self.process_depth(gpu_batch, processed_inputs, metadata)
        
        if 'stereo_T' in gpu_batch and gpu_batch['stereo_T'] is not None:
            processed_inputs['stereo_T'] = gpu_batch['stereo_T']
        
        return processed_inputs


    def move_to_gpu(self, batch_data):
        gpu_batch = {}
        
        for key, value in batch_data.items():
            if key == 'color_raw':
                gpu_batch[key] = {
                    frame_idx: tensor.to(self.device, non_blocking=True)
                    for frame_idx, tensor in value.items()
                }
            elif isinstance(value, torch.Tensor):
                gpu_batch[key] = value.to(self.device, non_blocking=True)
            else:
                gpu_batch[key] = value
                
        return gpu_batch
    
    
    def process_camera_params(self, gpu_batch, processed_inputs):
        K_raw = gpu_batch['K_raw']
        
        for scale in range(self.num_scales):
            K_scaled = K_raw.clone()
            
            scale_factor_w = self.width // (2 ** scale)
            scale_factor_h = self.height // (2 ** scale)
            
            K_scaled[:, 0, :] *= scale_factor_w
            K_scaled[:, 1, :] *= scale_factor_h
            
            processed_inputs[("K", scale)] = K_scaled
            processed_inputs[("inv_K", scale)] = torch.linalg.pinv(K_scaled)
    
    
    def process_depth(self, gpu_batch, processed_inputs, metadata):
        # depth_raw = gpu_batch['depth_raw']
        # batch_size = depth_raw.shape[0]
        
        # for b_idx in range(batch_size):
        #     if metadata['do_flip'][b_idx]:
        #         depth_raw[b_idx] = torch.flip(depth_raw[b_idx], dims=[2])
        
        # depth_gt = F.interpolate(
        #     depth_raw, 
        #     size=(self.height, self.width),
        #     mode='nearest'
        # )
        
        processed_inputs["depth_gt"] = gpu_batch['depth_raw']
        
        
# region - MonoDataset
class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'): #jpg
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # self.interp = Image.Resampling.LANCZOS

        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader

        self.to_tensor = transforms.ToTensor()
        self.load_depth = self.check_depth()
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

            
    # # region - preprocess
    # def preprocess(self, inputs, color_aug):
    #     """Resize colour images to the required scales and augment if required

    #     We create the color_aug object in advance and apply the same augmentation to all
    #     images in this item. This ensures that all images input to the pose network receive the
    #     same augmentation.
    #     """
    #     for k in list(inputs):
    #         frame = inputs[k]
    #         if "color" in k:
    #             n, im, i = k
    #             for i in range(self.num_scales):
    #                 inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

    #     for k in list(inputs):
    #         f = inputs[k]
    #         if "color" in k:
    #             n, im, i = k
    #             inputs[(n, im, i)] = self.to_tensor(f)
    #             inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))


    def __len__(self):
        return len(self.filenames)


    # region - getitem
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        line = self.filenames[index].split()
        folder = line[0]

        frame_index = int(line[1]) if len(line) == 3 else 0
        side        = line[2] if len(line) == 3 else None 

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        color_raw = {}
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                color = self.get_color(folder, frame_index, other_side, do_flip=False)
            else:
                color = self.get_color(folder, frame_index + i, side, do_flip=False)
                
            color_raw[i] = self.to_tensor(color)
            
        
        K_raw = torch.from_numpy(self.K.copy())

        depth_raw = None
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip=False)
            depth_raw = torch.from_numpy(np.expand_dims(depth_gt, 0).astype(np.float32))
            
        stereo_T = None
        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            stereo_T = torch.from_numpy(stereo_T)

        
        return {'color_raw': color_raw,
                'K_raw': K_raw,
                'depth_raw': depth_raw,
                'stereo_T': stereo_T,
                'do_color_aug': do_color_aug,
                'do_flip': do_flip,
                'folder': folder,
                'frame_index': frame_index,
                'side': side}


    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError


    def check_depth(self):
        raise NotImplementedError


    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
