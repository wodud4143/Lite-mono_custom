from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch
import torchvision.transforms.functional as TF
import kornia.geometry.transform as KTF


from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


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

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip, do_crop, crop_params,
                  do_scale_aug, scale_factor, do_translation_aug, translation_x, translation_y):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
            
        

        if do_crop and crop_params is not None:
            # 랜덤 크롭
            crop_x, crop_y, crop_w, crop_h = crop_params
            color = color.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        
        # Scale augmentation: PIL Image resize (개선 2)
        if do_scale_aug and scale_factor != 1.0:
            w, h = color.size
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            color = color.resize((new_w, new_h), pil.LANCZOS)
            # Resize 후 원래 크기로 복원 (crop 또는 padding)
            if scale_factor > 1.0:
                # Scale up: center crop
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                color = color.crop((left, top, left + w, top + h))
            else:
                # Scale down: padding
                new_color = pil.new('RGB', (w, h), (0, 0, 0))
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                new_color.paste(color, (left, top))
                color = new_color
        
        # Translation augmentation: PIL Image transform (개선 4)
        if do_translation_aug and (translation_x != 0.0 or translation_y != 0.0):
            w, h = color.size
            tx = int(translation_x * w)
            ty = int(translation_y * h)
            # Affine transformation matrix: [1, 0, tx, 0, 1, ty]
            color = color.transform(
                color.size,
                pil.AFFINE,
                (1, 0, tx, 0, 1, ty),
                fill=(0, 0, 0)
            )
        
        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str).replace("\\","/") #수정함
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        print(f_str)#디버깅용
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        print(image_path)#디버깅용
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
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
