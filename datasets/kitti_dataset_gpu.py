from __future__ import absolute_import, division, print_function

import os
import os.path as osp

import numpy as np
import PIL.Image as pil
import skimage.transform
import torch
from torch.utils.data import DataLoader


from kitti_utils import generate_depth_map
# from .mono_dataset import MonoDataset
from .mono_dataset_gpu import MonoDataset, GPUDataProcessor, collate_fn


# region - KITTIDataset
class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}


    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = osp.join(self.data_path, scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return osp.isfile(velo_filename)


    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if color.size != self.full_res_shape:
            color = color.resize(self.full_res_shape, pil.BILINEAR)
        
        # if do_flip:
        #     color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


# region - RAW Dataset
class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)


    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = osp.join(
            # self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str).replace("\\","/") #수정함
        return image_path


    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = osp.join(self.data_path, folder.split("/")[0])

        velo_filename = osp.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        # Velodyne 포인트 클라우드에서 뎁스 맵 생성
        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        # if do_flip:
        #     depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = osp.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


# region - Depth Dataset
class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)


    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = osp.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path


    def get_depth(self, folder, frame_index, side, do_flip):
    # def get_depth(self, folder, frame_index, side):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = osp.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


# region - Train pipeline
class KITTITrainingPipeline:
    def __init__(self, dataset_type='raw', data_path="", filenames_file="",
                 height=192, width=640, frame_idxs=[0, -1, 1], num_scales=4,
                 batch_size=8, num_workers=4, img_ext=".png", is_train=True, device='cuda'):
        
        self.device = torch.device(device)
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.is_train = is_train
        
        with open(filenames_file, 'r') as f:
            filenames = f.readlines()
        filenames = [line.strip() for line in filenames]
        
        dataset_classes = {
            'raw': KITTIRAWDataset,
            'odom': KITTIOdomDataset, 
            'depth': KITTIDepthDataset
        }
        
        dataset_class = dataset_classes[dataset_type]
        
        self.dataset = dataset_class(
            data_path=data_path,
            filenames=filenames,
            height=height,
            width=width,
            frame_idxs=frame_idxs,
            num_scales=num_scales,
            is_train=is_train
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=False, #pin_memory=True,
            collate_fn=collate_fn
            # persistent_workers=True if num_workers > 0 else False,
            # prefetch_factor=2 if num_workers > 0 else 2
        )
        
        self.gpu_processor = GPUDataProcessor(
            height=height,
            width=width,
            num_scales=num_scales,
            is_train=is_train,
            device=self.device
        )
    
    def get_dataloader(self):
        return self.dataloader
    
    def process_batch(self, batch_data):
        return self.gpu_processor.process_batch(batch_data)
    
    def __iter__(self):
        for batch_data in self.dataloader:
            yield self.process_batch(batch_data)
    
    def __len__(self):
        return len(self.dataloader)