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

    def get_color(self, folder, frame_index, side, do_flip, do_crop,crop_params):
    # def get_color(self, folder, frame_index, side, do_flip, do_crop,crop_params, do_cutout, cutout_params):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
            
        

        if do_crop and crop_params is not None:
            # 랜덤 크롭
            crop_x, crop_y, crop_w, crop_h = crop_params
            color = color.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            


        # if do_tr_aug:


        #     def pil_to_tensor(img):
        #         arr = np.array(img).astype(np.float32) / 255.0
        #         tensor = torch.from_numpy(arr).permute(2, 0, 1)
        #         return tensor

        #     def tensor_to_pil(tensor):
        #         arr = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        #         return pil.fromarray(arr)

        #     def translate_image(img_tensor, tx, ty):
        #         M = torch.tensor([
        #             [1., 0., tx],
        #             [0., 1., ty]
        #         ], dtype=torch.float32).unsqueeze(0)
        #         return KTF.warp_affine(img_tensor.unsqueeze(0), M,
        #                             dsize=(img_tensor.shape[1], img_tensor.shape[2]),
        #                             mode='nearest',
        #                             padding_mode='border',
        #                             align_corners=True).squeeze(0)

        
        #     color_tensor = pil_to_tensor(color)
        #     h, w = color_tensor.shape[1], color_tensor.shape[2]

    
        #     tx_ratio, ty_ratio = tr_params
        #     tx = int(tx_ratio * w)
        #     ty = int(ty_ratio * h)

        #     translated_tensor = translate_image(color_tensor, tx, ty)

        #     color = tensor_to_pil(translated_tensor)
        
        
        # if do_cutout and cutout_params is not None:
        #     # Cutout 적용
        #     color = self.apply_cutout(color, 
        #                             n_holes=cutout_params['n_holes'], 
        #                             length=cutout_params['length'])
                    

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
