# from __future__ import absolute_import, division, print_function

# import os
# import random
# import numpy as np
# import copy
# from PIL import Image,ImageEnhance    # using pillow-simd for increased speed

# import torch
# import torch.utils.data as data
# from torchvision import transforms


# def pil_loader(path):
#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             return img.convert('RGB')
        
        
# class SharpnessAugmentation:
#     """Sharpness augmentation class similar to ColorJitter"""
#     def __init__(self, sharpness=(0.5, 1.5)):
#         self.sharpness = sharpness
    
#     def __call__(self, img):
#         """Apply sharpness transformation to PIL Image"""
#         if isinstance(self.sharpness, (tuple, list)) and len(self.sharpness) == 2:
#             sharpness_factor = random.uniform(self.sharpness[0], self.sharpness[1])
#         else:
#             sharpness_factor = random.uniform(max(0, 1 - self.sharpness), 1 + self.sharpness)
        
#         enhancer = ImageEnhance.Sharpness(img)
#         return enhancer.enhance(sharpness_factor)


# class MonoDataset(data.Dataset):
#     """Superclass for monocular dataloaders

#     Args:
#         data_path
#         filenames
#         height
#         width
#         frame_idxs
#         num_scales
#         is_train
#         img_ext
#     """
#     def __init__(self,
#                  data_path,
#                  filenames,
#                  height,
#                  width,
#                  frame_idxs,
#                  num_scales,
#                  is_train=False,
#                 #  img_ext='.jpg'
#                  img_ext='.png'):
#         super(MonoDataset, self).__init__()

#         self.data_path = data_path
#         self.filenames = filenames
#         self.height = height
#         self.width = width
#         self.num_scales = num_scales
#         # self.interp = Image.ANTIALIAS
#         self.interp = Image.LANCZOS

#         self.frame_idxs = frame_idxs

#         self.is_train = is_train
#         self.img_ext = img_ext

#         self.loader = pil_loader
#         self.to_tensor = transforms.ToTensor()

#         # We need to specify augmentations differently in newer versions of torchvision.
#         # We first try the newer tuple version; if this fails we fall back to scalars
#         try:
#             self.brightness = (0.8, 1.3)
#             self.contrast = (0.8, 1.2)
#             self.saturation = (0.8, 1.2)
#             self.hue = (-0.1, 0.1)
#             transforms.ColorJitter.get_params(
#                 self.brightness, self.contrast, self.saturation, self.hue)
#         except TypeError:
#             self.brightness = 0.2
#             self.contrast = 0.2
#             self.saturation = 0.2
#             self.hue = 0.1

#         self.resize = {}
#         for i in range(self.num_scales):
#             s = 2 ** i
#             self.resize[i] = transforms.Resize((self.height // s, self.width // s),
#                                                interpolation=self.interp)

#         self.load_depth = self.check_depth()

#     def preprocess(self, inputs, color_aug, sharpness_aug):
#         """Resize colour images to the required scales and augment if required

#         We create the color_aug object in advance and apply the same augmentation to all
#         images in this item. This ensures that all images input to the pose network receive the
#         same augmentation.
#         """
#         for k in list(inputs):
#             frame = inputs[k]
#             if "color" in k:
#                 n, im, i = k
#                 for i in range(self.num_scales):
#                     inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

#         for k in list(inputs):
#             f = inputs[k]
#             if "color" in k:
#                 n, im, i = k
#                 inputs[(n, im, i)] = self.to_tensor(f)
                
#                 # Apply both color and sharpness augmentation
#                 augmented_img = color_aug(f)
#                 inputs[(n + "_aug", im, i)] = self.to_tensor(augmented_img)
                

        

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, index):
#         """Returns a single training item from the dataset as a dictionary.

#         Values correspond to torch tensors.
#         Keys in the dictionary are either strings or tuples:

#             ("color", <frame_id>, <scale>)          for raw colour images,
#             ("color_aug", <frame_id>, <scale>)      for augmented colour images,
#             ("K", scale) or ("inv_K", scale)        for camera intrinsics,
#             "stereo_T"                              for camera extrinsics, and
#             "depth_gt"                              for ground truth depth maps.

#         <frame_id> is either:
#             an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
#         or
#             "s" for the opposite image in the stereo pair.

#         <scale> is an integer representing the scale of the image relative to the fullsize image:
#             -1      images at native resolution as loaded from disk
#             0       images resized to (self.width,      self.height     )
#             1       images resized to (self.width // 2, self.height // 2)
#             2       images resized to (self.width // 4, self.height // 4)
#             3       images resized to (self.width // 8, self.height // 8)
#         """
#         inputs = {}

#         do_color_aug = self.is_train and random.random() > 0.5
#         do_flip = self.is_train and random.random() > 0.5


#         line = self.filenames[index].split()
#         folder = line[0]

#         if len(line) == 3:
#             frame_index = int(line[1])
#         else:
#             frame_index = 0

#         if len(line) == 3:
#             side = line[2]
#         else:
#             side = None

#         for i in self.frame_idxs:
#             if i == "s":
#                 other_side = {"r": "l", "l": "r"}[side]
#                 inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
#             else:
#                 inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

#         # adjusting intrinsics to match each scale in the pyramid
#         for scale in range(self.num_scales):
#             K = self.K.copy()

#             K[0, :] *= self.width // (2 ** scale)
#             K[1, :] *= self.height // (2 ** scale)

#             inv_K = np.linalg.pinv(K)

#             inputs[("K", scale)] = torch.from_numpy(K)
#             inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

#         if do_color_aug:
#             color_aug = transforms.ColorJitter(
#                 self.brightness, self.contrast, self.saturation, self.hue)
#         else:
#             color_aug = (lambda x: x)


            

#         self.preprocess(inputs, color_aug)


#         for i in self.frame_idxs:
#             del inputs[("color", i, -1)]
#             del inputs[("color_aug", i, -1)]

#         if self.load_depth:
#             depth_gt = self.get_depth(folder, frame_index, side, do_flip)
#             inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
#             inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

#         if "s" in self.frame_idxs:
#             stereo_T = np.eye(4, dtype=np.float32)
#             baseline_sign = -1 if do_flip else 1
#             side_sign = -1 if side == "l" else 1
#             stereo_T[0, 3] = side_sign * baseline_sign * 0.1

#             inputs["stereo_T"] = torch.from_numpy(stereo_T)

#         return inputs

#     def get_color(self, folder, frame_index, side, do_flip):
#         raise NotImplementedError

#     def check_depth(self):
#         raise NotImplementedError

#     def get_depth(self, folder, frame_index, side, do_flip):
#         raise NotImplementedError


from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# ==========================================================================================
# 🚀 Augmentation을 Torchvision Transform 스타일 클래스로 구현
# ==========================================================================================

# --- 🥇 최고의 조합(Best Combination)을 위한 클래스 ---
class RandomBlueHour(object):
    def __init__(self, intensity_range):
        self.intensity_range = intensity_range
    def __call__(self, img):
        intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
        # blue_hour 효과를 구성하는 하위 Augmentation 호출
        r, g, b = img.split(); r = r.point(lambda i: i * (1 - 0.15 * 0.6 * intensity)); b = b.point(lambda i: i * (1 + 0.15 * 0.6 * intensity)); img = Image.merge("RGB", (r, g, b))
        r, g, b = img.split(); b = b.point(lambda i: i * (1 + 0.15 * 0.2 * intensity)); img = Image.merge("RGB", (r, g, b))
        enhancer = ImageEnhance.Contrast(img); return enhancer.enhance(1.0 + (0.15 * intensity))

class RandomBlueTint(object):
    def __init__(self, intensity_range):
        self.intensity_range = intensity_range
    def __call__(self, img):
        intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
        r, g, b = img.split(); b = b.point(lambda i: i * (1 + 0.15 * intensity)); return Image.merge("RGB", (r, g, b))

class RandomBlur(object):
    def __init__(self, factor_range):
        self.factor_range = factor_range
    def __call__(self, img):
        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        enhancer = ImageEnhance.Sharpness(img); return enhancer.enhance(factor)

class RandomSaturation(object):
    def __init__(self, factor_range):
        self.factor_range = factor_range
    def __call__(self, img):
        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        enhancer = ImageEnhance.Color(img); return enhancer.enhance(factor)

# --- 🏋️‍♀️ 약점 보완(Weakness Training)을 위한 클래스 ---
class RandomPerspectiveTransform(object):
    def __init__(self, magnitude_range):
        self.magnitude_range = magnitude_range
    def __call__(self, img):
        magnitude = random.uniform(self.magnitude_range[0], self.magnitude_range[1])
        w, h = img.size; img_array = np.array(img); pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = magnitude * np.random.uniform(-1, 1, size=(4, 2)) * np.array([w, h]); pts2 = pts1 + offset
        M = cv2.getPerspectiveTransform(pts1, pts2.astype(np.float32))
        return Image.fromarray(cv2.warpPerspective(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE))

class RandomNoise(object):
    def __init__(self, noise_level_range):
        self.noise_level_range = noise_level_range
    def __call__(self, img):
        noise_level = random.uniform(self.noise_level_range[0], self.noise_level_range[1])
        img_array = np.array(img); noise = np.random.normal(0, noise_level, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255); return Image.fromarray(noisy_img.astype(np.uint8))

class RandomWarmTone(object):
    def __init__(self, intensity_range):
        self.intensity_range = intensity_range
    def __call__(self, img):
        intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
        r, g, b = img.split(); r = r.point(lambda i: i * (1 + 0.15 * intensity)); b = b.point(lambda i: i * (1 - 0.15 * intensity)); return Image.merge("RGB", (r, g, b))

class RandomHighlightRecovery(object):
    def __init__(self, intensity_range):
        self.intensity_range = intensity_range
    def __call__(self, img):
        intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
        gamma = 1.0 - (0.4 * intensity)
        if gamma <= 0: gamma = 1e-5
        table = np.array([((i / 255.0) ** (1/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return Image.fromarray(cv2.LUT(np.array(img), table))

# ==========================================================================================
# MonoDataset 클래스
# ==========================================================================================

class MonoDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        if self.is_train:
            core_augs = [
                transforms.RandomApply([RandomBlueHour(intensity_range=(0.2, 0.4))], p=0.5),
                transforms.RandomApply([RandomBlueTint(intensity_range=(0.2, 0.4))], p=0.5),
                transforms.RandomApply([RandomBlur(factor_range=(0.4, 0.5))], p=0.5),
                transforms.RandomApply([RandomSaturation(factor_range=(1.1, 1.3))], p=0.5),
            ]
            random.shuffle(core_augs)
            self.core_augmenter = transforms.Compose(core_augs)
            
            self.weakness_augmenter = transforms.RandomChoice([
                RandomNoise(noise_level_range=(2,8)),
                RandomWarmTone(intensity_range=(0.3, 0.5)),
                RandomHighlightRecovery(intensity_range=(0.2, 0.4)),
            ])
            
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                              interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, do_color_aug):
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i_scale in range(self.num_scales):
                    inputs[(n, im, i_scale)] = self.resize[i_scale](inputs[(n, im, i_scale - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                
                inputs[(n, im, i)] = self.to_tensor(f)

                augmented_img = f
                if do_color_aug and self.is_train:
                    augmented_img = self.core_augmenter(augmented_img)
                    
                    if random.random() < 0.5:
                        augmented_img = self.weakness_augmenter(augmented_img)
                
                inputs[(n + "_aug", im, i)] = self.to_tensor(augmented_img)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        frame_index = int(line[1]) if len(line) == 3 else 0
        side = line[2] if len(line) == 3 else None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        self.preprocess(inputs, do_color_aug)

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

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError