from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image,ImageDraw   # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


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
                #  img_ext='.jpg'
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # self.interp = Image.ANTIALIAS
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            # self.brightness = (0.8, 1.2)
            self.brightness = (0.8, 1.5)
            self.contrast = (0.7, 1.3)  # Contrast 강화: edge detection 향상 (개선 5)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        
    def apply_cutout(self, img, n_holes=1, length=100):
        img = img.copy()
        draw = ImageDraw.Draw(img)
        h, w = img.size[1], img.size[0]
        
        # 이미지의 아래쪽 절반 영역만 사용
        h_start = h // 2  
        
        for _ in range(n_holes):
            # y 좌표는 h_start부터 h까지의 범위에서만 선택
            # y = random.randint(h_start, h)
            y = random.randint(h_start + length // 2, h - length // 2)
            x = random.randint(0, w)
            
            y1 = max(h_start, y - length // 2)  # 최소값을 h_start로 제한
            y2 = min(h, y + length // 2)
            x1 = max(0, x - length // 2)
            x2 = min(w, x + length // 2)
            
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        
        return img

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                if n == "color":
                    inputs[(n, im, i)] = self.to_tensor(f)
                    inputs[("color_aug", im, i)] = self.to_tensor(color_aug(f))
        

    def __len__(self):
        return len(self.filenames)

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
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.3  # 50% → 70% (개선 5)
        do_flip = self.is_train and random.random() > 0.5
        do_scale_aug = self.is_train and random.random() > 0.5  # Scale augmentation (개선 2)
        do_translation_aug = self.is_train and random.random() > 0.5  # Translation augmentation (개선 4)
        do_crop = self.is_train and random.random() > 0.5
        # Cutout 제거: 시간 일관성 문제로 인해 제거 (개선 1)
        do_cutout = False


        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        if do_crop:
            #85-95% 크롭, 하단 중심 크롭 (개선 3)
            original_w, original_h = 1242, 375  
            crop_ratio = random.uniform(0.85, 0.95)
            crop_w = int(original_w * crop_ratio)
            crop_h = int(original_h * crop_ratio)
            crop_x = random.randint(0, original_w - crop_w)
            # 하단 중심 크롭: 하단 75% 영역에서만 선택 (가까운 객체 강화)
            crop_y = random.randint(
                max(0, original_h - crop_h - original_h // 4),  # 하단 75% 영역
                original_h - crop_h
            )
            crop_params = (crop_x, crop_y, crop_w, crop_h)
        else:
            crop_params = None
        
        # Scale augmentation parameters (모든 프레임에 동일하게 적용) (개선 2)
        if do_scale_aug:
            scale_factor = random.uniform(0.95, 1.05)  # 작은 범위로 시작
        else:
            scale_factor = 1.0
        
        # Translation augmentation parameters (모든 프레임에 동일하게 적용) (개선 4)
        if do_translation_aug:
            translation_x = random.uniform(-0.05, 0.05)  # 5% 범위
            translation_y = random.uniform(-0.05, 0.05)
        else:
            translation_x = 0.0
            translation_y = 0.0
            
        # if do_rotate:
        #     rot_angle =  random.choice([-1, 1]) * random.uniform(20, 25)
        # else:
        #     rot_angle = 0

        # if do_tr_aug:
        #     tr_params = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))  # 10%로 줄임
        # else:
        #     tr_params = (0, 0)

        # 모든 프레임에 동일한 augmentation 적용 (시간 일관성 보장)
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                # Scale과 Translation augmentation 적용 (모든 프레임에 동일) (개선 2, 4)
                color = self.get_color(folder, frame_index, other_side, do_flip, do_crop, crop_params, 
                                      do_scale_aug, scale_factor, do_translation_aug, translation_x, translation_y)
                inputs[("color", i, -1)] = color
            else:
                # Scale과 Translation augmentation 적용 (모든 프레임에 동일) (개선 2, 4)
                color = self.get_color(folder, frame_index + i, side, do_flip, do_crop, crop_params,
                                      do_scale_aug, scale_factor, do_translation_aug, translation_x, translation_y)
                inputs[("color", i, -1)] = color

        # adjusting intrinsics to match each scale in the pyramid
        # Geometry-preserving augmentation: Scale과 Translation에 대해 K 조정 (개선 2, 4)
        for scale in range(self.num_scales):
            K = self.K.copy()

            # Multi-scale에 대한 기본 조정
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            
            # Scale augmentation: Camera intrinsics 조정 (개선 2)
            if do_scale_aug:
                K[0, 0] *= scale_factor  # fx
                K[1, 1] *= scale_factor  # fy
                # Principal point 조정 (이미지 중심 기준)
                K[0, 2] = (K[0, 2] - self.width // (2 ** scale) / 2) * scale_factor + self.width // (2 ** scale) / 2  # cx
                K[1, 2] = (K[1, 2] - self.height // (2 ** scale) / 2) * scale_factor + self.height // (2 ** scale) / 2  # cy
            
            # Translation augmentation: Camera intrinsics 조정 (개선 4)
            if do_translation_aug:
                K[0, 2] += translation_x * self.width // (2 ** scale)  # cx
                K[1, 2] += translation_y * self.height // (2 ** scale)  # cy

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
            

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    
    def get_color(self, folder, frame_index, side, do_flip, do_crop, crop_params, 
                  do_scale_aug, scale_factor, do_translation_aug, translation_x, translation_y):
        raise NotImplementedError
    

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
