from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
import cv2
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
            self.brightness = (1.0, 1.0)  # brightness 비활성화 (Shadow/CLAHE 사용)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.0  # brightness 비활성화
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        
        # CLAHE 초기화
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

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
    
    def apply_clahe(self, img_np):
        """CLAHE 적용 (numpy array 입력, BGR 형식)"""
        lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    def apply_shadow(self, img_np, intensity_range=(0.5, 0.7)):
        """Shadow augmentation 적용 - Global Darkening (전체적으로 어둡게)
        단순 밝기 감소 (numpy array 입력, BGR 형식)
        """
        # 랜덤한 intensity 선택
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        
        # 전체 이미지에 일괄 적용
        out = img_np.astype(np.float32) * intensity
        return np.clip(out, 0, 255).astype(np.uint8)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k and "color_clahe" not in k and "color_shadow" not in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            elif "color_clahe" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            elif "color_shadow" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    
        for k in list(inputs):
            f = inputs[k]
            if "color" in k and "color_clahe" not in k and "color_shadow" not in k:
                n, im, i = k
                if n == "color":
                    inputs[(n, im, i)] = self.to_tensor(f) # 원본 이미지
            elif "color_clahe" in k:
                n, im, i = k
                if n == "color_clahe":
                    inputs[("color_aug", im, i)] = self.to_tensor(color_aug(f)) # CLAHE 적용된 이미지
            elif "color_shadow" in k:
                n, im, i = k
                if n == "color_shadow":
                    inputs[("color_aug_shadow", im, i)] = self.to_tensor(color_aug(f)) # Shadow 적용된 이미지

        

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

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        do_crop = self.is_train and random.random() > 0.5
        do_cutout = self.is_train and random.random() > 0.5


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
            #85-95% 크롭
            original_w, original_h = 1242, 375  
            crop_ratio = random.uniform(0.85, 0.95)
            crop_w = int(original_w * crop_ratio)
            crop_h = int(original_h * crop_ratio)
            crop_x = random.randint(0, original_w - crop_w)
            crop_y = random.randint(0, original_h - crop_h)
            crop_params = (crop_x, crop_y, crop_w, crop_h)
        else:
            crop_params = None
            
            
        if do_cutout:
            # Cutout parameters: 1-3 holes, size 50-150 pixels
            cutout_params = {
                'n_holes': 1,
                'length': random.randint(25, 50)
            }
        else:
            cutout_params = None
            

        # region CLAHE와 Shadow augmentation 적용
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                
                color_original, color_clahe, color_shadow = self.get_color(
                    folder, frame_index, other_side, do_flip, do_crop, crop_params, do_cutout, cutout_params)
                inputs[("color", i, -1)] = color_original
                inputs[("color_clahe", i, -1)] = color_clahe
                inputs[("color_shadow", i, -1)] = color_shadow

            else:
                color_original, color_clahe, color_shadow = self.get_color(
                    folder, frame_index + i, side, do_flip, do_crop, crop_params, do_cutout, cutout_params)
                inputs[("color", i, -1)] = color_original
                inputs[("color_clahe", i, -1)] = color_clahe
                inputs[("color_shadow", i, -1)] = color_shadow
               

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

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
            del inputs[("color_clahe", i, -1)]
            del inputs[("color_shadow", i, -1)]
            for scale in range(self.num_scales):
                if ("color_clahe", i, scale) in inputs:
                    del inputs[("color_clahe", i, scale)]
                if ("color_shadow", i, scale) in inputs:
                    del inputs[("color_shadow", i, scale)]

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

    
    def get_color(self, folder, frame_index, side, do_flip, do_crop, crop_params, do_cutout, cutout_params):
        """Returns (color_original, color_clahe, color_shadow) as PIL Images"""
        raise NotImplementedError
    

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
