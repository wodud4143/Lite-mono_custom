# from __future__ import absolute_import, division, print_function

# import os
# import sys
# import argparse
# import numpy as np
# import PIL.Image as pil
# from PIL import ImageEnhance, ImageFilter
# import cv2
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import itertools
# from tabulate import tabulate
# import json
# import time
# from tqdm import tqdm
# import importlib.util
# import random

# from layers import disp_to_depth
# from utils import readlines
# import datasets

# cv2.setNumThreads(0)
# splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# # --- Augmentation Parameter Space (Global for both search methods) ---
# AUGMENTATION_SPACE = {
#     'brightness': [{'intensity': v, 'factor': 1.1 + v*0.5} for v in np.linspace(0.2, 0.8, 4)],
#     'darkness': [{'intensity': v, 'factor': 0.9 - v*0.5} for v in np.linspace(0.2, 0.8, 4)],
#     'contrast': [{'intensity': v, 'factor': 1.1 + v*1.0} for v in np.linspace(0.2, 0.9, 5)],
#     'sharpness': [{'intensity': v, 'factor': 1.2 + v*1.8} for v in np.linspace(0.2, 0.8, 4)],
#     'blur': [{'intensity': v, 'factor': 0.8 - v*0.6} for v in np.linspace(0.2, 0.6, 3)],
#     'saturation': [{'intensity': v, 'factor': 1.1 + v*0.9} for v in np.linspace(0.2, 0.8, 4)],
#     'hue': [{'intensity': v, 'hue_shift': 5 + v*25} for v in np.linspace(0.3, 0.8, 4)],
#     'noise': [{'intensity': v, 'noise_level': 5 + v*25} for v in np.linspace(0.2, 0.7, 4)],
#     'gamma': [{'gamma': g} for g in [0.7, 0.9, 1.2, 1.5]],
#     'gaussian_blur': [{'intensity': v, 'radius': 0.5 + v*1.2} for v in np.linspace(0.3, 0.7, 3)],
#     'motion_blur': [{'intensity': v, 'size': 3 + v*4} for v in np.linspace(0.3, 0.6, 3)],
#     'channel_shift': [{'intensity': v, 'shift_range': 10 + v*20} for v in np.linspace(0.3, 0.6, 3)],
#     'warm_tone': [{'intensity': v} for v in [0.4, 0.7, 1.0]],
#     'cool_tone': [{'intensity': v} for v in [0.4, 0.7, 1.0]],
#     'yellow_tint': [{'intensity': v} for v in [0.5, 1.0]],
#     'blue_tint': [{'intensity': v} for v in [0.5, 1.0]],
#     'green_tint': [{'intensity': v} for v in [0.5, 1.0]],
#     'purple_tint': [{'intensity': v} for v in [0.5, 1.0]],
#     'high_exposure': [{'intensity': v} for v in [0.4, 0.8]],
#     'low_exposure': [{'intensity': v} for v in [0.3, 0.6]],
#     'shadow_boost': [{'intensity': v} for v in [0.4, 0.8]],
#     'highlight_recovery': [{'intensity': v} for v in [0.4, 0.8]],
#     'golden_hour': [{'intensity': v} for v in [0.5, 1.0]],
#     'blue_hour': [{'intensity': v} for v in [0.5, 1.0]],
#     'vignette': [{'intensity': v} for v in [0.6, 1.0]],
#     'desaturation': [{'intensity': v, 'factor': 0.4} for v in [0.5, 1.0]],
# }

# # --- Argument Parsing ---
# def parse_args():
#     parser = argparse.ArgumentParser(description='Intelligent augmentation search for depth estimation.')
    
#     # --- Search Strategy ---
#     parser.add_argument('--search_strategy', type=str, default='grid_search', choices=['grid_search', 'genetic'],
#                         help='Strategy to find the best augmentation. "grid_search" tests all predefined combinations, '
#                              '"genetic" uses a genetic algorithm for efficient search.')
    
#     # --- Genetic Algorithm Parameters ---
#     parser.add_argument('--population_size', type=int, default=50, help='Number of individuals in each generation for GA.')
#     parser.add_argument('--num_generations', type=int, default=15, help='Number of generations to evolve for GA.')
#     parser.add_argument('--mutation_rate', type=float, default=0.15, help='Probability of mutation for an individual in GA.')
#     parser.add_argument('--elitism_count', type=int, default=4, help='Number of best individuals to carry over to the next generation.')

#     # --- Other parameters (existing) ---
#     parser.add_argument('--data_path', type=str, default=r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\kitti_data", help='Path to KITTI dataset')
#     parser.add_argument('--model_name', type=str, default="v4_3_R_100", help='Model name')
#     parser.add_argument('--weights_folder', type=str, default='weights_99', help='Weights folder name')
#     parser.add_argument('--eval_split', type=str, default='eigen', choices=['eigen', 'eigen_benchmark', 'benchmark'], help='Evaluation split')
#     parser.add_argument('--model', type=str, default="lite-mono", choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"])
#     parser.add_argument("--no_cuda", action='store_true', help='Disable CUDA')
#     parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
#     parser.add_argument('--num_workers', type=int, default=12, help='Dataloader workers')
#     parser.add_argument('--save_results', type=str, default='augmentation_search_results.json', help='Path to save results')
#     parser.add_argument('--top_k', type=int, default=15, help='Number of top results to display')
#     parser.add_argument('--min_depth', type=float, default=0.1, help='Minimum depth')
#     parser.add_argument('--max_depth', type=float, default=100.0, help='Maximum depth')
#     parser.add_argument('--disable_median_scaling', action='store_true', help='Disable median scaling')
#     parser.add_argument('--pred_depth_scale_factor', type=float, default=1, help='Depth scale factor')
#     parser.add_argument('--quick_test', action='store_true', help='Use a small subset for quick testing')
#     parser.add_argument('--num_samples', type=int, default=-1, help='Number of test samples (-1 for all)')

#     return parser.parse_args()


# # --- Core Functions (Model Loading, Dataset, Evaluation) ---
# # ... (load_model_class, adjust_hue, AugmentedKITTIDataset, apply_augmentation_with_params, compute_errors, evaluate_augmentation functions are unchanged) ...
# def load_model_class(file_path, class_name):
#     """Dynamically load a class from a Python file"""
#     spec = importlib.util.spec_from_file_location("module", file_path)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[f"module_{file_path}"] = module
#     spec.loader.exec_module(module)
#     return getattr(module, class_name)


# def adjust_hue(image, hue_shift):
#     """Adjust hue of PIL image"""
#     img_array = np.array(image)
#     hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
#     hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
#     rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
#     return pil.fromarray(rgb)


# class AugmentedKITTIDataset(datasets.KITTIRAWDataset):
#     """Extended KITTI dataset with augmentation support"""
    
#     def __init__(self, *args, augmentation_list=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.augmentation_list = augmentation_list
    
#     def apply_augmentations(self, image):
#         if self.augmentation_list is None: return image
#         result = image
#         for aug_type, params in self.augmentation_list:
#             result = apply_augmentation_with_params(result, aug_type, params)
#         return result
    
#     def __getitem__(self, index):
#         inputs = super().__getitem__(index)
#         if self.augmentation_list is not None:
#             for k in list(inputs.keys()):
#                 if k == ("color", 0, 0):
#                     img_tensor = inputs[k]
#                     img_pil = transforms.ToPILImage()(img_tensor)
#                     img_augmented = self.apply_augmentations(img_pil)
#                     inputs[k] = transforms.ToTensor()(img_augmented)
#         return inputs


# def apply_augmentation_with_params(image, aug_type, params):
#     # This function remains the same as in the previous version
#     # It contains the logic for all 14+ augmentations
#     intensity = params.get('intensity', 1.0)
#     img_array = np.array(image)
#     if aug_type == 'brightness':
#         factor = params.get('factor', 1.3)
#         enhancer = ImageEnhance.Brightness(image)
#         return enhancer.enhance(1.0 + (factor - 1.0) * intensity)
#     elif aug_type == 'darkness':
#         factor = params.get('factor', 0.7)
#         enhancer = ImageEnhance.Brightness(image)
#         return enhancer.enhance(factor + (1.0 - factor) * (1.0 - intensity))
#     elif aug_type == 'contrast':
#         factor = params.get('factor', 1.5)
#         enhancer = ImageEnhance.Contrast(image)
#         return enhancer.enhance(1.0 + (factor - 1.0) * intensity)
#     elif aug_type == 'sharpness':
#         factor = params.get('factor', 2.0)
#         enhancer = ImageEnhance.Sharpness(image)
#         return enhancer.enhance(1.0 + (factor - 1.0) * intensity)
#     elif aug_type == 'blur':
#         factor = params.get('factor', 0.3)
#         enhancer = ImageEnhance.Sharpness(image)
#         return enhancer.enhance(factor + (1.0 - factor) * (1.0 - intensity))
#     elif aug_type == 'saturation':
#         factor = params.get('factor', 1.5)
#         enhancer = ImageEnhance.Color(image)
#         return enhancer.enhance(1.0 + (factor - 1.0) * intensity)
#     elif aug_type == 'hue':
#         hue_shift = params.get('hue_shift', 10) # degrees
#         hue_shift_cv = int((hue_shift * intensity * 179) / 360)
#         return adjust_hue(image, hue_shift_cv)
#     elif aug_type == 'noise':
#         noise_level = params.get('noise_level', 20)
#         noise = np.random.normal(0, noise_level * intensity, img_array.shape)
#         noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
#         return pil.fromarray(noisy_img)
#     elif aug_type == 'gamma':
#         gamma = params.get('gamma', 1.5)
#         img_array_float = img_array.astype(np.float32) / 255.0
#         corrected = np.power(img_array_float, gamma)
#         corrected = (corrected * 255).astype(np.uint8)
#         return pil.fromarray(corrected)
#     elif aug_type == 'gaussian_blur':
#         radius = params.get('radius', 1.0)
#         return image.filter(ImageFilter.GaussianBlur(radius=radius * intensity))
#     elif aug_type == 'motion_blur':
#         size = int(params.get('size', 5) * intensity)
#         if size > 1:
#             kernel = np.zeros((size, size))
#             kernel[int((size - 1) / 2), :] = np.ones(size)
#             kernel = kernel / size
#             blurred = cv2.filter2D(img_array, -1, kernel)
#             return pil.fromarray(blurred)
#         return image
#     elif aug_type == 'channel_shift':
#         shift_range = params.get('shift_range', 20)
#         shifts = np.random.uniform(-shift_range, shift_range, 3) * intensity
#         img_array_float = img_array.astype(np.float32)
#         for i in range(3):
#             img_array_float[:, :, i] += shifts[i]
#         img_array_float = np.clip(img_array_float, 0, 255).astype(np.uint8)
#         return pil.fromarray(img_array_float)
#     elif aug_type == 'warm_tone':
#         r, g, b = image.split()
#         r = r.point(lambda i: i * (1 + 0.15 * intensity))
#         b = b.point(lambda i: i * (1 - 0.15 * intensity))
#         return pil.merge("RGB", (r, g, b))
#     elif aug_type == 'cool_tone':
#         r, g, b = image.split()
#         r = r.point(lambda i: i * (1 - 0.15 * intensity))
#         b = b.point(lambda i: i * (1 + 0.15 * intensity))
#         return pil.merge("RGB", (r, g, b))
#     elif aug_type == 'yellow_tint':
#         r, g, b = image.split()
#         r = r.point(lambda i: i * (1 + 0.1 * intensity))
#         g = g.point(lambda i: i * (1 + 0.1 * intensity))
#         return pil.merge("RGB", (r, g, b))
#     elif aug_type == 'blue_tint':
#         r, g, b = image.split()
#         b = b.point(lambda i: i * (1 + 0.15 * intensity))
#         return pil.merge("RGB", (r, g, b))
#     elif aug_type == 'green_tint':
#         r, g, b = image.split()
#         g = g.point(lambda i: i * (1 + 0.15 * intensity))
#         return pil.merge("RGB", (r, g, b))
#     elif aug_type == 'purple_tint':
#         r, g, b = image.split()
#         r = r.point(lambda i: i * (1 + 0.1 * intensity))
#         b = b.point(lambda i: i * (1 + 0.1 * intensity))
#         return pil.merge("RGB", (r, g, b))
#     elif aug_type == 'high_exposure':
#         gamma = 1.0 - (0.4 * intensity)
#         inv_gamma = 1.0 / gamma if gamma > 0 else 1.0
#         table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#         return pil.fromarray(cv2.LUT(img_array, table))
#     elif aug_type == 'low_exposure':
#         gamma = 1.0 + (0.6 * intensity)
#         inv_gamma = 1.0 / gamma
#         table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#         return pil.fromarray(cv2.LUT(img_array, table))
#     elif aug_type == 'shadow_boost':
#         gamma = 1.0 + (0.5 * intensity)
#         table = np.array([((i / 255.0) ** (1/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
#         return pil.fromarray(cv2.LUT(img_array, table))
#     elif aug_type == 'highlight_recovery':
#         gamma = 1.0 - (0.4 * intensity)
#         table = np.array([((i / 255.0) ** (1/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
#         return pil.fromarray(cv2.LUT(img_array, table))
#     elif aug_type == 'golden_hour':
#         img = apply_augmentation_with_params(image, 'warm_tone', {'intensity': 0.5 * intensity})
#         img = apply_augmentation_with_params(img, 'yellow_tint', {'intensity': 0.3 * intensity})
#         enhancer = ImageEnhance.Contrast(img)
#         return enhancer.enhance(1.0 - (0.15 * intensity))
#     elif aug_type == 'blue_hour':
#         img = apply_augmentation_with_params(image, 'cool_tone', {'intensity': 0.6 * intensity})
#         img = apply_augmentation_with_params(img, 'blue_tint', {'intensity': 0.2 * intensity})
#         enhancer = ImageEnhance.Contrast(img)
#         return enhancer.enhance(1.0 + (0.15 * intensity))
#     elif aug_type == 'vignette':
#         rows, cols = img_array.shape[:2]
#         kernel_x = cv2.getGaussianKernel(cols, int(200 * intensity))
#         kernel_y = cv2.getGaussianKernel(rows, int(200 * intensity))
#         kernel = kernel_y * kernel_x.T
#         mask = 255 * kernel / np.linalg.norm(kernel)
#         mask = (mask - mask.min()) / (mask.max() - mask.min())
#         output = img_array.astype(np.float32)
#         for i in range(3):
#             output[:,:,i] *= mask
#         return pil.fromarray(np.clip(output, 0, 255).astype(np.uint8))
#     elif aug_type == 'desaturation':
#         factor = params.get('factor', 0.4)
#         enhancer = ImageEnhance.Color(image)
#         return enhancer.enhance(1.0 - (1.0 - factor) * intensity)
#     else:
#         return image

# def compute_errors(gt, pred):
#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()
#     rmse = np.sqrt(((gt - pred) ** 2).mean())
#     rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#     sq_rel = np.mean(((gt - pred) ** 2) / gt)
#     return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# def evaluate_augmentation(encoder, depth_decoder, dataloader, gt_depths, args, aug_name=""):
#     MIN_DEPTH, MAX_DEPTH = 1e-3, 80
#     encoder.eval()
#     depth_decoder.eval()
#     pred_disps = []
#     with torch.no_grad():
#         for data in tqdm(dataloader, desc=f"Evaluating {aug_name}", leave=False, ncols=100):
#             input_color = data[("color", 0, 0)].cuda() if not args.no_cuda else data[("color", 0, 0)]
#             output = depth_decoder(encoder(input_color))
#             pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
#             pred_disps.append(pred_disp.cpu()[:, 0].numpy())
#     pred_disps = np.concatenate(pred_disps)
#     errors = []
#     num_samples = min(len(pred_disps), len(gt_depths))
#     for i in range(num_samples):
#         gt_depth = gt_depths[i]
#         gt_height, gt_width = gt_depth.shape[:2]
#         pred_disp = cv2.resize(pred_disps[i], (gt_width, gt_height))
#         pred_depth = 1 / pred_disp
#         mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
#         if args.eval_split == "eigen":
#             crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
#                              0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
#             crop_mask = np.zeros(mask.shape)
#             crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
#             mask &= crop_mask.astype(bool)
#         pred_depth, gt_depth = pred_depth[mask], gt_depth[mask]
#         pred_depth *= args.pred_depth_scale_factor
#         if not args.disable_median_scaling:
#             ratio = np.median(gt_depth) / np.median(pred_depth)
#             pred_depth *= ratio
#         pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)
#         errors.append(compute_errors(gt_depth, pred_depth))
#     return np.array(errors).mean(0)

# # --- Grid Search Strategy ---
# def generate_grid_search_combinations(quick_test=False):
#     # This is the original generate_augmentation_combinations function
#     # ... (code is unchanged from previous version) ...
#     aug_params = AUGMENTATION_SPACE
#     if quick_test:
#         for key in aug_params: aug_params[key] = aug_params[key][:1]
#     combinations = [{'name': 'Original', 'augs': []}]
#     for aug_type, params_list in aug_params.items():
#         for i, params in enumerate(params_list):
#             combinations.append({'name': f'{aug_type}_{i}', 'augs': [(aug_type, params)]})
#     important_2_combos = [('brightness', 'contrast'), ('contrast', 'sharpness'), ('darkness', 'contrast'), ('saturation', 'hue'), ('warm_tone', 'contrast'), ('cool_tone', 'sharpness'), ('high_exposure', 'saturation'), ('shadow_boost', 'saturation'), ('golden_hour', 'sharpness'), ('vignette', 'contrast'), ('gaussian_blur', 'noise')]
#     for aug1, aug2 in important_2_combos:
#         if aug1 in aug_params and aug2 in aug_params:
#             for i in [0, 1]:
#                 if i < len(aug_params[aug1]) and i < len(aug_params[aug2]):
#                     p1, p2 = aug_params[aug1][i], aug_params[aug2][i]
#                     combinations.append({'name': f'{aug1}+{aug2}_{i}', 'augs': [(aug1, p1), (aug2, p2)]})
#     important_3_combos = [['brightness', 'contrast', 'saturation'], ['darkness', 'contrast', 'noise'], ['contrast', 'sharpness', 'saturation'], ['warm_tone', 'contrast', 'vignette'], ['shadow_boost', 'contrast', 'saturation'], ['golden_hour', 'contrast', 'sharpness']]
#     for combo in important_3_combos:
#         augs, valid = [], True
#         for aug in combo:
#             if aug in aug_params and 0 < len(aug_params[aug]): augs.append((aug, aug_params[aug][0]))
#             else: valid = False; break
#         if valid: combinations.append({'name': f"{'+'.join(combo)}", 'augs': augs})
#     special_combos = [{'name': 'Golden_Hour_Pro', 'augs': [('golden_hour', {'intensity': 0.8}), ('warm_tone', {'intensity': 0.6}), ('yellow_tint', {'intensity': 0.5})]}, {'name': 'Blue_Hour_Pro', 'augs': [('blue_hour', {'intensity': 0.8}), ('cool_tone', {'intensity': 0.7}), ('blue_tint', {'intensity': 0.6})]}, {'name': 'HDR_Style', 'augs': [('contrast', {'intensity': 0.7, 'factor': 1.6}), ('saturation', {'intensity': 0.5, 'factor': 1.4}), ('shadow_boost', {'intensity': 0.5})]}, {'name': 'Night_Vision', 'augs': [('darkness', {'intensity': 0.7, 'factor': 0.5}), ('green_tint', {'intensity': 1.0}), ('contrast', {'intensity': 0.6, 'factor': 1.5}), ('noise', {'intensity': 0.6, 'noise_level': 20})]}, {'name': 'Fog_Simulation', 'augs': [('gaussian_blur', {'intensity': 0.5, 'radius': 1.0}), ('brightness', {'intensity': 0.4, 'factor': 1.2}), ('contrast', {'intensity': 0.4, 'factor': 0.7}), ('desaturation', {'intensity': 0.8, 'factor': 0.3})]}, {'name': 'Rain_Simulation', 'augs': [('motion_blur', {'intensity': 0.4, 'size': 4}), ('darkness', {'intensity': 0.5, 'factor': 0.7}), ('cool_tone', {'intensity': 0.6}), ('contrast', {'intensity': 0.4, 'factor': 1.3})]}]
#     combinations.extend(special_combos)
#     return combinations

# # --- 🤖 Genetic Algorithm Strategy ---
# def create_random_individual():
#     """Creates a random augmentation pipeline (a chromosome)."""
#     num_augs = random.randint(1, 4)  # Pipeline will have 1 to 4 augmentations
#     individual_augs = []
#     available_augs = list(AUGMENTATION_SPACE.keys())
    
#     for _ in range(num_augs):
#         aug_type = random.choice(available_augs)
#         params = random.choice(AUGMENTATION_SPACE[aug_type])
#         individual_augs.append((aug_type, params))
        
#     # Remove duplicates to ensure a clean pipeline
#     # Note: This simple approach might slightly bias towards shorter pipelines if many duplicates are picked
#     unique_augs = []
#     seen_types = set()
#     for aug_type, params in individual_augs:
#         if aug_type not in seen_types:
#             unique_augs.append((aug_type, params))
#             seen_types.add(aug_type)

#     return unique_augs

# def initialize_population(size):
#     """Creates the initial population of random individuals."""
#     return [create_random_individual() for _ in range(size)]

# def crossover(parent1, parent2):
#     """Performs single-point crossover between two parents."""
#     if not parent1 or not parent2:
#         return parent1 or parent2
    
#     point = random.randint(1, min(len(parent1), len(parent2)))
#     child_augs = parent1[:point] + parent2[point:]
    
#     # Clean up duplicates that may arise from crossover
#     unique_augs = []
#     seen_types = set()
#     for aug_type, params in child_augs:
#         if aug_type not in seen_types:
#             unique_augs.append((aug_type, params))
#             seen_types.add(aug_type)
#     return unique_augs

# def mutate(individual, mutation_rate):
#     """Applies mutation to an individual."""
#     if random.random() > mutation_rate or not individual:
#         return individual

#     mutated_individual = individual[:]
#     mutation_type = random.choice(['change_param', 'add_aug', 'remove_aug', 'swap_aug'])

#     if mutation_type == 'change_param' and mutated_individual:
#         idx_to_mutate = random.randrange(len(mutated_individual))
#         aug_type, _ = mutated_individual[idx_to_mutate]
#         new_params = random.choice(AUGMENTATION_SPACE[aug_type])
#         mutated_individual[idx_to_mutate] = (aug_type, new_params)

#     elif mutation_type == 'add_aug' and len(mutated_individual) < 5:
#         available_augs = list(set(AUGMENTATION_SPACE.keys()) - {aug[0] for aug in mutated_individual})
#         if available_augs:
#             new_aug_type = random.choice(available_augs)
#             new_params = random.choice(AUGMENTATION_SPACE[new_aug_type])
#             mutated_individual.append((new_aug_type, new_params))

#     elif mutation_type == 'remove_aug' and len(mutated_individual) > 1:
#         mutated_individual.pop(random.randrange(len(mutated_individual)))

#     elif mutation_type == 'swap_aug' and mutated_individual:
#         idx_to_swap = random.randrange(len(mutated_individual))
#         current_aug_type, _ = mutated_individual[idx_to_swap]
#         available_augs = list(set(AUGMENTATION_SPACE.keys()) - {aug[0] for aug in mutated_individual} | {current_aug_type})
#         if available_augs:
#             new_aug_type = random.choice(available_augs)
#             new_params = random.choice(AUGMENTATION_SPACE[new_aug_type])
#             mutated_individual[idx_to_swap] = (new_aug_type, new_params)
            
#     return mutated_individual

# # --- Main Execution Logic ---
# def main():
#     args = parse_args()
    
#     print("=" * 80)
#     print("🚀 Advanced Augmentation Search for Depth Estimation v3 🚀")
#     print(f"🧬 Search Strategy: {args.search_strategy.upper()}")
#     print("=" * 80)
    
#     # --- Setup model and data (same as before) ---
#     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#     model_name, weights_folder = args.model_name, args.weights_folder
#     base_path = f"experiments/logs/{model_name}"
#     encoder_path = os.path.join(base_path, "models", weights_folder, "encoder.pth")
#     decoder_path = os.path.join(base_path, "models", weights_folder, "depth.pth")
#     encoder_file_path = os.path.join(base_path, f"{model_name}_encoder.py")
#     decoder_file_path = os.path.join(base_path, f"{model_name}_decoder.py")
    
#     print(f"-> Loading model: {model_name} from {weights_folder}")
#     LiteMono = load_model_class(encoder_file_path, "LiteMono")
#     DepthDecoder = load_model_class(decoder_file_path, "DepthDecoder")
    
#     encoder_dict = torch.load(encoder_path, map_location=device)
#     decoder_dict = torch.load(decoder_path, map_location=device)
    
#     encoder = LiteMono(model=args.model, height=encoder_dict['height'], width=encoder_dict['width'])
#     depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    
#     encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
#     depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_decoder.state_dict()})
    
#     encoder.to(device).eval()
#     depth_decoder.to(device).eval()
    
#     filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
#     if args.num_samples > 0: filenames = filenames[:args.num_samples]
    
#     gt_path = os.path.join(splits_dir, args.eval_split, "gt_depths.npz")
#     gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
#     if args.num_samples > 0: gt_depths = gt_depths[:args.num_samples]

#     all_time_results = []
#     memoization = {} # Cache results to avoid re-evaluation
    
#     def get_fitness(individual_augs):
#         # Create a unique key for the augmentation pipeline
#         individual_key = tuple(sorted((aug[0], str(aug[1])) for aug in individual_augs))
        
#         if individual_key in memoization:
#             return memoization[individual_key]

#         name = "+".join([aug[0] for aug in individual_augs]) or "Original"
        
#         dataset = AugmentedKITTIDataset(
#             args.data_path, filenames, encoder_dict['height'], encoder_dict['width'],
#             [0], 4, is_train=False, augmentation_list=individual_augs if individual_augs else None)
        
#         dataloader = DataLoader(
#             dataset, args.batch_size, shuffle=False, num_workers=0 if os.name == 'nt' else args.num_workers,
#             pin_memory=True, drop_last=False)
        
#         try:
#             mean_errors = evaluate_augmentation(encoder, depth_decoder, dataloader, gt_depths, args, name)
#             abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors
#             result = {'name': name, 'augs': individual_augs, 'abs_rel': float(abs_rel), 'sq_rel': float(sq_rel),
#                       'rmse': float(rmse), 'rmse_log': float(rmse_log), 'a1': float(a1), 'a2': float(a2), 'a3': float(a3)}
#             all_time_results.append(result)
            
#             # Fitness is the inverse of the weighted error score (lower error = higher fitness)
#             weighted_error = result['sq_rel'] * 2 + result['abs_rel']
#             fitness = 1 / weighted_error if weighted_error > 0 else float('inf')
            
#             memoization[individual_key] = (result, fitness)
#             return result, fitness
            
#         except Exception as e:
#             print(f"   ERROR during evaluation for {name}: {e}")
#             return None, 0

#     start_time = time.time()
    
#     if args.search_strategy == 'grid_search':
#         print("\n-> Generating predefined combinations for Grid Search...")
#         combinations = generate_grid_search_combinations(quick_test=args.quick_test)
#         print(f"   Total combinations to test: {len(combinations)}")
#         print("\n-> Starting Grid Search Evaluation...")
#         for combo in combinations:
#             get_fitness(combo['augs']) # Evaluate and populate all_time_results

#     elif args.search_strategy == 'genetic':
#         print("\n-> Initializing population for Genetic Algorithm...")
#         population = initialize_population(args.population_size)
#         # Ensure 'Original' is always evaluated
#         population[0] = [] 
        
#         for gen in range(args.num_generations):
#             print(f"\n--- Generation {gen + 1}/{args.num_generations} ---")
            
#             # Evaluate population and get fitness scores
#             eval_results = [get_fitness(ind) for ind in population]
            
#             # Filter out failed evaluations and sort by fitness (descending)
#             population_with_fitness = sorted(
#                 [(ind, res, fit) for ind, (res, fit) in zip(population, eval_results) if res is not None],
#                 key=lambda x: x[2], reverse=True
#             )
            
#             if not population_with_fitness:
#                 print("   Evaluation failed for all individuals. Stopping.")
#                 break

#             best_ind_res = population_with_fitness[0][1]
#             print(f"   Best of Gen: {best_ind_res['name'][:50]} | "
#                   f"sq_rel: {best_ind_res['sq_rel']:.4f}, abs_rel: {best_ind_res['abs_rel']:.4f}")

#             # Create next generation
#             next_generation = []
            
#             # Elitism: carry over the best individuals
#             elites = [ind for ind, _, _ in population_with_fitness[:args.elitism_count]]
#             next_generation.extend(elites)
            
#             # Crossover and Mutation
#             while len(next_generation) < args.population_size:
#                 # Tournament selection
#                 p1 = random.choice(population_with_fitness)[0]
#                 p2 = random.choice(population_with_fitness)[0]
#                 parent1 = p1 if get_fitness(p1)[1] > get_fitness(p2)[1] else p2
                
#                 p3 = random.choice(population_with_fitness)[0]
#                 p4 = random.choice(population_with_fitness)[0]
#                 parent2 = p3 if get_fitness(p3)[1] > get_fitness(p4)[1] else p4

#                 child = crossover(parent1, parent2)
#                 child = mutate(child, args.mutation_rate)
#                 next_generation.append(child)
            
#             population = next_generation

#     elapsed_time = time.time() - start_time
#     print(f"\n-> Search completed in {elapsed_time/60:.2f} minutes")
#     print(f"-> Total unique combinations evaluated: {len(all_time_results)}")

#     # --- Final Analysis (same for both strategies) ---
#     if not all_time_results:
#         print("\nNo results to analyze. Exiting.")
#         return

#     all_time_results.sort(key=lambda x: (x['sq_rel'] * 2 + x['abs_rel']))
    
#     print("\n" + "=" * 140)
#     print(f"🏆 TOP {args.top_k} AUGMENTATION COMBINATIONS (sq_rel-focused)")
#     print("🎯 Primary: sq_rel (제곱 상대 오차) 💎💎 | Secondary: abs_rel (절대 상대 오차) ⭐")
#     print("=" * 140)
    
#     table_data = []
#     for i, r in enumerate(all_time_results[:args.top_k]):
#         weighted_score = r['sq_rel'] * 2 + r['abs_rel']
#         table_data.append([
#             i + 1, r['name'][:40], f"{weighted_score:.4f}",
#             f"{r['sq_rel']:.4f}", f"{r['abs_rel']:.4f}", f"{r['rmse']:.3f}",
#             f"{r['rmse_log']:.4f}", f"{r['a1']:.3f}", f"{r['a2']:.3f}", f"{r['a3']:.3f}"
#         ])
    
#     headers = ["Rank", "Augmentation", "Weighted↓", "sq_rel↓💎💎", "abs_rel↓⭐", "rmse↓", "rmse_log↓", "a1↑", "a2↑", "a3↑"]
#     print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
#     # ... (The rest of the analysis and saving part is the same as the previous version) ...
#     print("\n" + "=" * 80)
#     print("🥇 BEST AUGMENTATION DETAILS")
#     print("=" * 80)
#     best = all_time_results[0]
#     best_weighted_score = best['sq_rel'] * 2 + best['abs_rel']
#     print(f"Name: {best['name']}")
#     if best['augs']:
#         print("Parameters:")
#         for aug_type, params in best['augs']: print(f"   - {aug_type}: {params}")
    
#     print(f"\n🎯 핵심 성능 지표:")
#     print(f"   Weighted Score (sq_rel*2 + abs_rel): {best_weighted_score:.4f} 🏆")
#     print(f"   sq_rel (제곱 상대 오차):  {best['sq_rel']:.4f} 💎💎")
#     print(f"   abs_rel (절대 상대 오차): {best['abs_rel']:.4f} ⭐")
    
#     original = next((r for r in all_time_results if r['name'] == 'Original'), None)
#     if original and best['name'] != 'Original':
#         print("\n" + "-" * 80)
#         print("🚀 IMPROVEMENT OVER ORIGINAL")
#         print("-" * 80)
        
#         original_weighted = original['sq_rel'] * 2 + original['abs_rel']
#         improvement = (original_weighted - best_weighted_score) / original_weighted * 100
#         sq_rel_improvement = (original['sq_rel'] - best['sq_rel']) / original['sq_rel'] * 100
#         abs_rel_improvement = (original['abs_rel'] - best['abs_rel']) / original['abs_rel'] * 100
        
#         print(f"🏆 종합 가중 점수 개선: {original_weighted:.4f} → {best_weighted_score:.4f} (↓{improvement:.2f}%)")
#         print(f"💎 sq_rel (Primary):  {original['sq_rel']:.4f} → {best['sq_rel']:.4f} (↓{sq_rel_improvement:.2f}%)")
#         print(f"⭐ abs_rel (Secondary): {original['abs_rel']:.4f} → {best['abs_rel']:.4f} (↓{abs_rel_improvement:.2f}%)")
    
#     if args.save_results:
#         save_data = {
#             'search_strategy': args.search_strategy,
#             'model_name': args.model_name,
#             'weights_folder': args.weights_folder,
#             'settings': {
#                 'eval_split': args.eval_split,
#                 'num_test_images': len(filenames),
#                 'total_unique_evals': len(all_time_results),
#             },
#             'best_combination': best,
#             'results_sorted_by_weighted_score': all_time_results
#         }
#         with open(args.save_results, 'w') as f: json.dump(save_data, f, indent=2)
#         print(f"\n-> Enhanced results saved to {args.save_results}")

#     print("\n" + "=" * 80)
#     print("🎉 ADVANCED AUGMENTATION SEARCH COMPLETE! 🎉")
#     print("=" * 80)


# if __name__ == '__main__':
#     main()


from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import numpy as np
import PIL.Image as pil
from PIL import ImageEnhance, ImageFilter, ImageOps
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tabulate import tabulate
import json
import time
from tqdm import tqdm
import importlib.util
import random
import io

from layers import disp_to_depth
from utils import readlines
import datasets

cv2.setNumThreads(0)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# --- Augmentation Parameter Space (v4 - 8 New Augs Added) ---
AUGMENTATION_SPACE = {
    # Color & Light
    'brightness': [{'intensity': v, 'factor': 1.1 + v*0.5} for v in np.linspace(0.2, 0.8, 4)],
    'darkness': [{'intensity': v, 'factor': 0.9 - v*0.5} for v in np.linspace(0.2, 0.8, 4)],
    'contrast': [{'intensity': v, 'factor': 1.1 + v*1.0} for v in np.linspace(0.2, 0.9, 5)],
    'saturation': [{'intensity': v, 'factor': 1.1 + v*0.9} for v in np.linspace(0.2, 0.8, 4)],
    'hue': [{'intensity': v, 'hue_shift': 5 + v*25} for v in np.linspace(0.3, 0.8, 4)],
    'gamma': [{'gamma': g} for g in [0.7, 0.9, 1.2, 1.5]],
    'warm_tone': [{'intensity': v} for v in [0.4, 0.7, 1.0]],
    'cool_tone': [{'intensity': v} for v in [0.4, 0.7, 1.0]],
    'yellow_tint': [{'intensity': v} for v in [0.5, 1.0]],
    'blue_tint': [{'intensity': v} for v in [0.5, 1.0]],
    'green_tint': [{'intensity': v} for v in [0.5, 1.0]],
    'purple_tint': [{'intensity': v} for v in [0.5, 1.0]],
    'high_exposure': [{'intensity': v} for v in [0.4, 0.8]],
    'low_exposure': [{'intensity': v} for v in [0.3, 0.6]],
    'shadow_boost': [{'intensity': v} for v in [0.4, 0.8]],
    'highlight_recovery': [{'intensity': v} for v in [0.4, 0.8]],
    'channel_shift': [{'intensity': v, 'shift_range': 10 + v*20} for v in np.linspace(0.3, 0.6, 3)],
    # Blur, Noise & Sharpness
    'sharpness': [{'intensity': v, 'factor': 1.2 + v*1.8} for v in np.linspace(0.2, 0.8, 4)],
    'blur': [{'intensity': v, 'factor': 0.8 - v*0.6} for v in np.linspace(0.2, 0.6, 3)],
    'noise': [{'intensity': v, 'noise_level': 5 + v*25} for v in np.linspace(0.2, 0.7, 4)],
    'gaussian_blur': [{'intensity': v, 'radius': 0.5 + v*1.2} for v in np.linspace(0.3, 0.7, 3)],
    'motion_blur': [{'intensity': v, 'size': 3 + v*4} for v in np.linspace(0.3, 0.6, 3)],
    # Special Effects
    'golden_hour': [{'intensity': v} for v in [0.5, 1.0]],
    'blue_hour': [{'intensity': v} for v in [0.5, 1.0]],
    'vignette': [{'intensity': v} for v in [0.6, 1.0]],
    'desaturation': [{'intensity': v, 'factor': 0.4} for v in [0.5, 1.0]],
    
    # 📸 NEW Geometric Distortions
    'perspective_transform': [{'magnitude': m} for m in [0.05, 0.1]],
    'affine_transform': [{'rotation': r, 'translate_percent': t, 'scale': s, 'shear': sh} for r, t, s, sh in [(5, 0.05, 0.95, 3), (10, 0.1, 0.9, 5)]],
    'elastic_transform': [{'alpha': a, 'sigma': s} for a, s in [(34, 4), (50, 5)]],
    
    # 🌍 NEW Real-world Corruptions
    'jpeg_compression': [{'quality': q} for q in [25, 15, 10]],
    'chromatic_aberration': [{'shift_amount': s} for s in [1, 2]],
    
    # ⚫ NEW Pixel-level Manipulations
    'cutout': [{'num_holes': n, 'hole_size': s} for n, s in [(8, 20), (12, 25)]],
    'posterize': [{'bits': b} for b in [3, 2]],
    'solarize': [{'threshold': t} for t in [128, 96]],
}

def parse_args():
    parser = argparse.ArgumentParser(description='Intelligent augmentation search for depth estimation.')
    parser.add_argument('--search_strategy', type=str, default='genetic', choices=['grid_search', 'genetic'], help='Strategy to find the best augmentation.')
    parser.add_argument('--population_size', type=int, default=30, help='Number of individuals in each generation for GA.')
    parser.add_argument('--num_generations', type=int, default=15, help='Number of generations to evolve for GA.')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='Probability of mutation for an individual in GA.')
    parser.add_argument('--elitism_count', type=int, default=3, help='Number of best individuals to carry over.')
    parser.add_argument('--data_path', type=str, default=r".\kitti_data", help='Path to KITTI dataset')
    parser.add_argument('--model_name', type=str, default="v4_3_R_100", help='Model name')
    parser.add_argument('--weights_folder', type=str, default='weights_99', help='Weights folder name')
    parser.add_argument('--eval_split', type=str, default='eigen', choices=['eigen', 'eigen_benchmark', 'benchmark'], help='Evaluation split')
    parser.add_argument('--model', type=str, default="lite-mono", choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"])
    parser.add_argument("--no_cuda", action='store_true', help='Disable CUDA')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='Dataloader workers')
    parser.add_argument('--save_results', type=str, default='augmentation_search_results.json', help='Path to save results')
    parser.add_argument('--top_k', type=int, default=15, help='Number of top results to display')
    parser.add_argument('--min_depth', type=float, default=0.1, help='Minimum depth')
    parser.add_argument('--max_depth', type=float, default=100.0, help='Maximum depth')
    parser.add_argument('--disable_median_scaling', action='store_true', help='Disable median scaling')
    parser.add_argument('--pred_depth_scale_factor', type=float, default=1, help='Depth scale factor')
    parser.add_argument('--quick_test', action='store_true', help='Use a small subset for quick testing')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of test samples (-1 for all)')
    return parser.parse_args()

def load_model_class(file_path, class_name):
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"module_{file_path}"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def adjust_hue(image, hue_shift):
    img_array = np.array(image)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return pil.fromarray(rgb)

class AugmentedKITTIDataset(datasets.KITTIRAWDataset):
    def __init__(self, *args, augmentation_list=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_list = augmentation_list
    def apply_augmentations(self, image):
        if self.augmentation_list is None: return image
        result = image
        for aug_type, params in self.augmentation_list:
            result = apply_augmentation_with_params(result, aug_type, params)
        return result
    def __getitem__(self, index):
        inputs = super().__getitem__(index)
        if self.augmentation_list is not None:
            for k in list(inputs.keys()):
                if k == ("color", 0, 0):
                    img_tensor = inputs[k]
                    img_pil = transforms.ToPILImage()(img_tensor)
                    img_augmented = self.apply_augmentations(img_pil)
                    inputs[k] = transforms.ToTensor()(img_augmented)
        return inputs

def apply_augmentation_with_params(image, aug_type, params):
    intensity = params.get('intensity', 1.0)
    img_array = np.array(image)

    # Color & Light
    if aug_type == 'brightness':
        factor = params.get('factor', 1.3); enhancer = ImageEnhance.Brightness(image); return enhancer.enhance(1.0 + (factor - 1.0) * intensity)
    elif aug_type == 'darkness':
        factor = params.get('factor', 0.7); enhancer = ImageEnhance.Brightness(image); return enhancer.enhance(factor + (1.0 - factor) * (1.0 - intensity))
    elif aug_type == 'contrast':
        factor = params.get('factor', 1.5); enhancer = ImageEnhance.Contrast(image); return enhancer.enhance(1.0 + (factor - 1.0) * intensity)
    elif aug_type == 'saturation':
        factor = params.get('factor', 1.5); enhancer = ImageEnhance.Color(image); return enhancer.enhance(1.0 + (factor - 1.0) * intensity)
    elif aug_type == 'hue':
        hue_shift = params.get('hue_shift', 10); hue_shift_cv = int((hue_shift * intensity * 179) / 360); return adjust_hue(image, hue_shift_cv)
    elif aug_type == 'gamma':
        gamma = params.get('gamma', 1.5); img_float = np.array(image, dtype=np.float32) / 255.0; corrected = np.power(img_float, gamma); return pil.fromarray((corrected * 255).astype(np.uint8))
    elif aug_type == 'warm_tone':
        r, g, b = image.split(); r = r.point(lambda i: i * (1 + 0.15 * intensity)); b = b.point(lambda i: i * (1 - 0.15 * intensity)); return pil.merge("RGB", (r, g, b))
    elif aug_type == 'cool_tone':
        r, g, b = image.split(); r = r.point(lambda i: i * (1 - 0.15 * intensity)); b = b.point(lambda i: i * (1 + 0.15 * intensity)); return pil.merge("RGB", (r, g, b))
    elif aug_type == 'yellow_tint':
        r, g, b = image.split(); r = r.point(lambda i: i * (1 + 0.1 * intensity)); g = g.point(lambda i: i * (1 + 0.1 * intensity)); return pil.merge("RGB", (r, g, b))
    elif aug_type == 'blue_tint':
        r, g, b = image.split(); b = b.point(lambda i: i * (1 + 0.15 * intensity)); return pil.merge("RGB", (r, g, b))
    elif aug_type == 'green_tint':
        r, g, b = image.split(); g = g.point(lambda i: i * (1 + 0.15 * intensity)); return pil.merge("RGB", (r, g, b))
    elif aug_type == 'purple_tint':
        r, g, b = image.split(); r = r.point(lambda i: i * (1 + 0.1 * intensity)); b = b.point(lambda i: i * (1 + 0.1 * intensity)); return pil.merge("RGB", (r, g, b))
    elif aug_type == 'high_exposure':
        gamma = 1.0 - (0.4 * intensity); inv_gamma = 1.0 / gamma if gamma > 0 else 1.0; table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8"); return pil.fromarray(cv2.LUT(img_array, table))
    elif aug_type == 'low_exposure':
        gamma = 1.0 + (0.6 * intensity); inv_gamma = 1.0 / gamma; table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8"); return pil.fromarray(cv2.LUT(img_array, table))
    elif aug_type == 'shadow_boost':
        gamma = 1.0 + (0.5 * intensity); table = np.array([((i / 255.0) ** (1/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8"); return pil.fromarray(cv2.LUT(img_array, table))
    elif aug_type == 'highlight_recovery':
        gamma = 1.0 - (0.4 * intensity); table = np.array([((i / 255.0) ** (1/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8"); return pil.fromarray(cv2.LUT(img_array, table))
    elif aug_type == 'channel_shift':
        shift_range = params.get('shift_range', 20); shifts = np.random.uniform(-shift_range, shift_range, 3) * intensity; img_float = img_array.astype(np.float32)
        for i in range(3): img_float[:, :, i] += shifts[i]
        return pil.fromarray(np.clip(img_float, 0, 255).astype(np.uint8))
    
    # Blur, Noise & Sharpness
    elif aug_type == 'sharpness':
        factor = params.get('factor', 2.0); enhancer = ImageEnhance.Sharpness(image); return enhancer.enhance(1.0 + (factor - 1.0) * intensity)
    elif aug_type == 'blur':
        factor = params.get('factor', 0.3); enhancer = ImageEnhance.Sharpness(image); return enhancer.enhance(factor + (1.0 - factor) * (1.0 - intensity))
    elif aug_type == 'noise':
        noise_level = params.get('noise_level', 20); noise = np.random.normal(0, noise_level * intensity, img_array.shape); noisy_img = np.clip(img_array + noise, 0, 255); return pil.fromarray(noisy_img.astype(np.uint8))
    elif aug_type == 'gaussian_blur':
        radius = params.get('radius', 1.0); return image.filter(ImageFilter.GaussianBlur(radius=radius * intensity))
    elif aug_type == 'motion_blur':
        size = int(params.get('size', 5) * intensity)
        if size > 1: kernel = np.zeros((size, size)); kernel[int((size - 1) / 2), :] = np.ones(size); kernel = kernel / size; return pil.fromarray(cv2.filter2D(img_array, -1, kernel))
        return image
        
    # Special Effects
    elif aug_type == 'golden_hour':
        img = apply_augmentation_with_params(image, 'warm_tone', {'intensity': 0.5 * intensity}); img = apply_augmentation_with_params(img, 'yellow_tint', {'intensity': 0.3 * intensity}); enhancer = ImageEnhance.Contrast(img); return enhancer.enhance(1.0 - (0.15 * intensity))
    elif aug_type == 'blue_hour':
        img = apply_augmentation_with_params(image, 'cool_tone', {'intensity': 0.6 * intensity}); img = apply_augmentation_with_params(img, 'blue_tint', {'intensity': 0.2 * intensity}); enhancer = ImageEnhance.Contrast(img); return enhancer.enhance(1.0 + (0.15 * intensity))
    elif aug_type == 'vignette':
        rows, cols = img_array.shape[:2]; kernel_x = cv2.getGaussianKernel(cols, int(200*intensity)); kernel_y = cv2.getGaussianKernel(rows, int(200*intensity)); kernel = kernel_y * kernel_x.T
        mask = (kernel - kernel.min()) / (kernel.max() - kernel.min()); output = img_array.astype(np.float32)
        for i in range(3): output[:,:,i] *= mask
        return pil.fromarray(np.clip(output, 0, 255).astype(np.uint8))
    elif aug_type == 'desaturation':
        factor = params.get('factor', 0.4); enhancer = ImageEnhance.Color(image); return enhancer.enhance(1.0 - (1.0 - factor) * intensity)
    
    # Geometric Distortions
    elif aug_type == 'perspective_transform':
        magnitude = params.get('magnitude', 0.1); w, h = image.size; pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = magnitude * np.random.uniform(-1, 1, size=(4, 2)) * np.array([w, h]); pts2 = pts1 + offset
        M = cv2.getPerspectiveTransform(pts1, pts2.astype(np.float32)); return pil.fromarray(cv2.warpPerspective(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    elif aug_type == 'affine_transform':
        angle = params.get('rotation', 0) * random.uniform(-1, 1); tx = params.get('translate_percent', 0) * image.size[0] * random.uniform(-1, 1)
        ty = params.get('translate_percent', 0) * image.size[1] * random.uniform(-1, 1); scale = random.uniform(params.get('scale', 1.0), 1.0)
        shear = params.get('shear', 0) * random.uniform(-1, 1)
        return image.transform(image.size, pil.AFFINE, (1/scale, shear, -tx, 0, 1/scale, -ty), resample=pil.BILINEAR, fillcolor=(128,128,128))
    elif aug_type == 'elastic_transform':
        alpha = params.get('alpha', 34); sigma = params.get('sigma', 4); shape = img_array.shape
        dx = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), sigma) * alpha; dy = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0])); map_x, map_y = (x + dx).astype(np.float32), (y + dy).astype(np.float32)
        return pil.fromarray(cv2.remap(img_array, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
    
    # Real-world Corruptions
    elif aug_type == 'jpeg_compression':
        quality = params.get('quality', 25); buffer = io.BytesIO(); image.save(buffer, "JPEG", quality=quality); buffer.seek(0); return pil.open(buffer)
    elif aug_type == 'chromatic_aberration':
        shift = params.get('shift_amount', 1); r, g, b = image.split(); r_s = np.roll(r, (shift, shift), axis=(0, 1)); b_s = np.roll(b, (-shift, -shift), axis=(0, 1)); return pil.merge("RGB", (pil.fromarray(r_s), g, pil.fromarray(b_s)))
    
    # Pixel-level Manipulations
    elif aug_type == 'cutout':
        num_holes = params.get('num_holes', 8); size = params.get('hole_size', 20); h, w, _ = img_array.shape; img_cutout = img_array.copy()
        for _ in range(num_holes):
            y = np.random.randint(h); x = np.random.randint(w); y1=np.clip(y-size//2,0,h);y2=np.clip(y+size//2,0,h);x1=np.clip(x-size//2,0,w);x2=np.clip(x+size//2,0,w); img_cutout[y1:y2,x1:x2]=128
        return pil.fromarray(img_cutout)
    elif aug_type == 'posterize':
        return ImageOps.posterize(image, params.get('bits', 3))
    elif aug_type == 'solarize':
        return ImageOps.solarize(image, params.get('threshold', 128))
    
    return image

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt)); a1 = (thresh < 1.25).mean(); a2 = (thresh < 1.25 ** 2).mean(); a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean()); rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt); sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate_augmentation(encoder, depth_decoder, dataloader, gt_depths, args, aug_name=""):
    MIN_DEPTH, MAX_DEPTH = 1e-3, 80; encoder.eval(); depth_decoder.eval(); pred_disps = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Evaluating {aug_name}", leave=False, ncols=100):
            input_color = data[("color", 0, 0)].cuda() if not args.no_cuda else data[("color", 0, 0)]
            output = depth_decoder(encoder(input_color)); pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
            pred_disps.append(pred_disp.cpu()[:, 0].numpy())
    pred_disps = np.concatenate(pred_disps); errors = []; num_samples = min(len(pred_disps), len(gt_depths))
    for i in range(num_samples):
        gt_depth = gt_depths[i]; gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = cv2.resize(pred_disps[i], (gt_width, gt_height)); pred_depth = 1 / pred_disp
        mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
        if args.eval_split == "eigen":
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape); crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1; mask &= crop_mask.astype(bool)
        pred_depth, gt_depth = pred_depth[mask], gt_depth[mask]
        pred_depth *= args.pred_depth_scale_factor
        if not args.disable_median_scaling:
            if np.median(pred_depth) > 0: ratio = np.median(gt_depth) / np.median(pred_depth); pred_depth *= ratio
        pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH); errors.append(compute_errors(gt_depth, pred_depth))
    return np.array(errors).mean(0)

def generate_grid_search_combinations(quick_test=False):
    aug_params = AUGMENTATION_SPACE
    if quick_test:
        for key in aug_params: aug_params[key] = aug_params[key][:1]
    combinations = [{'name': 'Original', 'augs': []}]
    for aug_type, params_list in aug_params.items():
        for i, params in enumerate(params_list):
            combinations.append({'name': f'{aug_type}_{i}', 'augs': [(aug_type, params)]})
    important_2_combos = [('brightness', 'contrast'), ('contrast', 'sharpness'), ('darkness', 'contrast'), ('saturation', 'hue'), ('warm_tone', 'contrast'), ('cool_tone', 'sharpness'), ('high_exposure', 'saturation'), ('shadow_boost', 'saturation'), ('golden_hour', 'sharpness'), ('vignette', 'contrast'), ('gaussian_blur', 'noise')]
    for aug1, aug2 in important_2_combos:
        if aug1 in aug_params and aug2 in aug_params:
            for i in [0, 1]:
                if i < len(aug_params[aug1]) and i < len(aug_params[aug2]):
                    p1, p2 = aug_params[aug1][i], aug_params[aug2][i]
                    combinations.append({'name': f'{aug1}+{aug2}_{i}', 'augs': [(aug1, p1), (aug2, p2)]})
    important_3_combos = [['brightness', 'contrast', 'saturation'], ['darkness', 'contrast', 'noise'], ['contrast', 'sharpness', 'saturation'], ['warm_tone', 'contrast', 'vignette'], ['shadow_boost', 'contrast', 'saturation'], ['golden_hour', 'contrast', 'sharpness']]
    for combo in important_3_combos:
        augs, valid = [], True
        for aug in combo:
            if aug in aug_params and 0 < len(aug_params[aug]): augs.append((aug, aug_params[aug][0]))
            else: valid = False; break
        if valid: combinations.append({'name': f"{'+'.join(combo)}", 'augs': augs})
    return combinations

def create_random_individual():
    num_augs = random.randint(1, 4); individual_augs = []
    available_augs = list(AUGMENTATION_SPACE.keys())
    for _ in range(num_augs):
        aug_type = random.choice(available_augs)
        params = random.choice(AUGMENTATION_SPACE[aug_type])
        individual_augs.append((aug_type, params))
    unique_augs = []; seen_types = set()
    for aug_type, params in individual_augs:
        if aug_type not in seen_types:
            unique_augs.append((aug_type, params)); seen_types.add(aug_type)
    return unique_augs

def initialize_population(size): return [create_random_individual() for _ in range(size)]

def crossover(p1, p2):
    if not p1 or not p2: return p1 or p2
    if len(p1) == 1 or len(p2) == 1: pt = 1
    else: pt = random.randint(1, min(len(p1), len(p2)))
    child = p1[:pt] + p2[pt:]
    unique = []; seen = set()
    for at, p in child:
        if at not in seen: unique.append((at, p)); seen.add(at)
    return unique

def mutate(ind, rate):
    if random.random() > rate or not ind: return ind
    mut_ind = ind[:]; mtype = random.choice(['change_param', 'add_aug', 'remove_aug', 'swap_aug'])
    if mtype == 'change_param':
        idx = random.randrange(len(mut_ind)); at, _ = mut_ind[idx]; p = random.choice(AUGMENTATION_SPACE[at]); mut_ind[idx] = (at,p)
    elif mtype == 'add_aug' and len(mut_ind) < 5:
        avail = list(set(AUGMENTATION_SPACE.keys()) - {a[0] for a in mut_ind})
        if avail: at = random.choice(avail); p = random.choice(AUGMENTATION_SPACE[at]); mut_ind.append((at,p))
    elif mtype == 'remove_aug' and len(mut_ind) > 1:
        mut_ind.pop(random.randrange(len(mut_ind)))
    elif mtype == 'swap_aug':
        idx = random.randrange(len(mut_ind)); current_at, _ = mut_ind[idx]
        avail = list(set(AUGMENTATION_SPACE.keys()) - {a[0] for a in mut_ind} | {current_at})
        if avail: at = random.choice(avail); p = random.choice(AUGMENTATION_SPACE[at]); mut_ind[idx] = (at,p)
    return mut_ind

def main():
    args = parse_args()
    print("=" * 80); print("🚀 Advanced Augmentation Search for Depth Estimation v5 🚀"); print("     (Weakness Analysis & Training Recommendations Added)"); print(f"🧬 Search Strategy: {args.search_strategy.upper()}"); print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model_name, weights_folder = args.model_name, args.weights_folder; base_path = f"experiments/logs/{model_name}"
    encoder_path = os.path.join(base_path, "models", weights_folder, "encoder.pth"); decoder_path = os.path.join(base_path, "models", weights_folder, "depth.pth")
    encoder_file_path = os.path.join(base_path, f"{model_name}_encoder.py"); decoder_file_path = os.path.join(base_path, f"{model_name}_decoder.py")
    
    print(f"-> Loading model: {model_name} from {weights_folder}")
    LiteMono = load_model_class(encoder_file_path, "LiteMono"); DepthDecoder = load_model_class(decoder_file_path, "DepthDecoder")
    encoder_dict = torch.load(encoder_path, map_location=device); decoder_dict = torch.load(decoder_path, map_location=device)
    encoder = LiteMono(model=args.model, height=encoder_dict['height'], width=encoder_dict['width']); depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_decoder.state_dict()})
    encoder.to(device).eval(); depth_decoder.to(device).eval()
    
    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
    if args.num_samples > 0: filenames = filenames[:args.num_samples]
    gt_path = os.path.join(splits_dir, args.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    if args.num_samples > 0: gt_depths = gt_depths[:args.num_samples]

    all_time_results = []; memoization = {}

    def get_fitness(individual_augs):
        individual_key = tuple(sorted((aug[0], str(aug[1])) for aug in individual_augs))
        if individual_key in memoization: return memoization[individual_key]
        name = "+".join([aug[0] for aug in individual_augs]) or "Original"
        dataset = AugmentedKITTIDataset(args.data_path, filenames, encoder_dict['height'], encoder_dict['width'],[0], 4, is_train=False, augmentation_list=individual_augs or None)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0 if os.name == 'nt' else args.num_workers, pin_memory=True, drop_last=False)
        try:
            mean_errors = evaluate_augmentation(encoder, depth_decoder, dataloader, gt_depths, args, name)
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors
            result = {'name': name, 'augs': individual_augs, 'abs_rel': float(abs_rel), 'sq_rel': float(sq_rel),'rmse': float(rmse), 'rmse_log': float(rmse_log), 'a1': float(a1), 'a2': float(a2), 'a3': float(a3)}
            all_time_results.append(result)
            weighted_error = result['sq_rel'] * 2 + result['abs_rel']
            fitness = 1 / weighted_error if weighted_error > 0 else float('inf')
            memoization[individual_key] = (result, fitness)
            return result, fitness
        except Exception as e:
            print(f"   ERROR during evaluation for {name}: {e}"); return None, 0

    start_time = time.time()
    if args.search_strategy == 'genetic':
        population = initialize_population(args.population_size); population[0] = [] 
        for gen in range(args.num_generations):
            print(f"\n--- Generation {gen + 1}/{args.num_generations} ---")
            eval_results = [get_fitness(ind) for ind in population]
            population_with_fitness = sorted([(ind, res, fit) for ind, (res, fit) in zip(population, eval_results) if res is not None], key=lambda x: x[2], reverse=True)
            if not population_with_fitness: print("   Evaluation failed. Stopping."); break
            best_ind_res = population_with_fitness[0][1]
            print(f"   Best of Gen: {best_ind_res['name'][:50]} | sq_rel: {best_ind_res['sq_rel']:.4f}, abs_rel: {best_ind_res['abs_rel']:.4f}")
            next_generation = [ind for ind, _, _ in population_with_fitness[:args.elitism_count]]
            while len(next_generation) < args.population_size:
                p1 = random.choice(population_with_fitness)[0]; p2 = random.choice(population_with_fitness)[0]
                parent1 = p1 if get_fitness(p1)[1] > get_fitness(p2)[1] else p2
                p3 = random.choice(population_with_fitness)[0]; p4 = random.choice(population_with_fitness)[0]
                parent2 = p3 if get_fitness(p3)[1] > get_fitness(p4)[1] else p4
                child = crossover(parent1, parent2); child = mutate(child, args.mutation_rate)
                next_generation.append(child)
            population = next_generation
    
    elapsed_time = time.time() - start_time
    print(f"\n-> Search completed in {elapsed_time/60:.2f} minutes")
    print(f"-> Total unique combinations evaluated: {len(all_time_results)}")

    if not all_time_results: print("\nNo results to analyze. Exiting."); return
    
    for r in all_time_results: r['weighted_score'] = r['sq_rel'] * 2 + r['abs_rel']
    all_time_results.sort(key=lambda x: x['weighted_score'])
    
    print("\n" + "=" * 140)
    print(f"🏆 TOP {args.top_k} AUGMENTATION COMBINATIONS (Best Performers)")
    print("🎯 Primary: sq_rel (제곱 상대 오차) 💎💎 | Secondary: abs_rel (절대 상대 오차) ⭐")
    table_data = []
    for i, r in enumerate(all_time_results[:args.top_k]):
        table_data.append([i + 1, r['name'][:40], f"{r['weighted_score']:.4f}", f"{r['sq_rel']:.4f}", f"{r['abs_rel']:.4f}", f"{r['rmse']:.3f}", f"{r['rmse_log']:.4f}", f"{r['a1']:.3f}", f"{r['a2']:.3f}", f"{r['a3']:.3f}"])
    headers = ["Rank", "Augmentation", "Weighted↓", "sq_rel↓💎💎", "abs_rel↓⭐", "rmse↓", "rmse_log↓", "a1↑", "a2↑", "a3↑"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\n" + "=" * 80); print("🥇 BEST AUGMENTATION DETAILS"); print("=" * 80)
    best = all_time_results[0]
    print(f"Name: {best['name']}")
    if best['augs']:
        print("Parameters:"); [print(f"   - {aug_type}: {params}") for aug_type, params in best['augs']]
    print(f"\n🎯 핵심 성능 지표:"); print(f"   Weighted Score: {best['weighted_score']:.4f} 🏆"); print(f"   sq_rel:  {best['sq_rel']:.4f} 💎💎"); print(f"   abs_rel: {best['abs_rel']:.4f} ⭐")
    
    original = next((r for r in all_time_results if r['name'] == 'Original'), None)
    if original and best['name'] != 'Original':
        print("\n" + "-" * 80); print("🚀 IMPROVEMENT OVER ORIGINAL"); print("-" * 80)
        improvement = (original['weighted_score'] - best['weighted_score']) / original['weighted_score'] * 100
        sq_rel_improvement = (original['sq_rel'] - best['sq_rel']) / original['sq_rel'] * 100
        abs_rel_improvement = (original['abs_rel'] - best['abs_rel']) / original['abs_rel'] * 100
        print(f"🏆 Weighted Score: {original['weighted_score']:.4f} → {best['weighted_score']:.4f} (↓{improvement:.2f}%)")
        print(f"💎 sq_rel:  {original['sq_rel']:.4f} → {best['sq_rel']:.4f} (↓{sq_rel_improvement:.2f}%)")
        print(f"⭐ abs_rel: {original['abs_rel']:.4f} → {best['abs_rel']:.4f} (↓{abs_rel_improvement:.2f}%)")
        
    if original:
        print("\n" + "=" * 140); print(f"🧐 弱点分析及び改善提案 (Weakness Analysis & Improvement Suggestions)"); print("=" * 140)
        worst_performers = sorted([r for r in all_time_results if r['name'] != 'Original'], key=lambda x: x['weighted_score'], reverse=True)
        print("\n📉 가장 성능 하락이 큰 Augmentation 조합 (Top 5 Worst Combinations)"); print("-" * 80)
        table_data_worst = []
        for r in worst_performers[:5]:
            degradation = ((r['weighted_score'] - original['weighted_score']) / original['weighted_score'] * 100)
            table_data_worst.append([r['name'][:50], f"{r['weighted_score']:.4f}", f"{original['weighted_score']:.4f}", f"🔻 {degradation:.1f}%"])
        headers_worst = ["Augmentation", "Weighted Score (높을수록 나쁨)", "Original Score", "성능 하락률"]
        print(tabulate(table_data_worst, headers=headers_worst, tablefmt="grid"))
        
        single_aug_results = [r for r in all_time_results if len(r['augs']) == 1]
        degradation_by_type = {}
        for r in single_aug_results:
            aug_type = r['augs'][0][0]
            if aug_type not in degradation_by_type: degradation_by_type[aug_type] = []
            degradation = ((r['weighted_score'] - original['weighted_score']) / original['weighted_score'])
            degradation_by_type[aug_type].append(degradation)
        avg_degradation = []
        for aug_type, scores in degradation_by_type.items():
            if scores: avg_degradation.append((aug_type, np.mean(scores) * 100))
        avg_degradation.sort(key=lambda x: x[1], reverse=True)
        
        print("\n📊 Augmentation 타입별 평균 성능 하락률 (Higher is worse)"); print("-" * 80)
        for aug_type, avg_deg in avg_degradation[:10]: print(f"  - {aug_type:<25}: 🔻 {avg_deg:.2f}%")
        
        print("\n🎯 다음 학습을 위한 추천 (Recommendations for Next Training)"); print("-" * 80)
        print("모델의 강건성을 향상시키기 위해, 다음 훈련 시 아래 Augmentation들을 학습 파이프라인에 **확률적으로** 추가하는 것을 강력히 권장합니다:")
        recommended_augs = [aug[0] for aug in avg_degradation[:4]]
        for i, aug in enumerate(recommended_augs): print(f"  {i+1}. **{aug}**")
        print("=" * 140)

    if args.save_results:
        save_data = {'search_strategy': args.search_strategy, 'model_name': args.model_name, 'settings': {'total_unique_evals': len(all_time_results)}, 'best_combination': best, 'results_sorted_by_weighted_score': all_time_results}
        with open(args.save_results, 'w') as f: json.dump(save_data, f, indent=2)
        print(f"\n-> Enhanced results saved to {args.save_results}")

    print("\n" + "=" * 80); print("🎉 ADVANCED AUGMENTATION SEARCH COMPLETE! 🎉"); print("=" * 80)
    
    
    # ============================================================================
    # 🎨 상위 1위 Augmentation 시각화 및 저장
    # ============================================================================
    test_image_path = "test.png"
    if os.path.exists(test_image_path) and best['augs']:
        print("\n" + "=" * 80)
        print("🎨 Visualizing Best Augmentation on test.png")
        print("=" * 80)
        
        # 원본 이미지 로드
        original_img = pil.open(test_image_path).convert('RGB')
        
        # 모델 입력 크기로 리사이즈
        feed_height = encoder_dict['height']
        feed_width = encoder_dict['width']
        original_resized = original_img.resize((feed_width, feed_height), pil.LANCZOS)
        
        # 1위 augmentation 적용
        augmented_img = original_resized
        for aug_type, params in best['augs']:
            augmented_img = apply_augmentation_with_params(augmented_img, aug_type, params)
        
        # Tensor 변환 함수
        to_tensor = transforms.ToTensor()
        
        # 원본 추론
        with torch.no_grad():
            input_original = to_tensor(original_resized).unsqueeze(0).to(device)
            output_original = depth_decoder(encoder(input_original))
            disp_original = output_original[("disp", 0)]
            disp_original_np = disp_original.squeeze().cpu().numpy()
            
            # 변환된 이미지 추론
            input_augmented = to_tensor(augmented_img).unsqueeze(0).to(device)
            output_augmented = depth_decoder(encoder(input_augmented))
            disp_augmented = output_augmented[("disp", 0)]
            disp_augmented_np = disp_augmented.squeeze().cpu().numpy()
        
        # Depth map을 시각화 (normalize to 0-255)
        def normalize_depth(depth):
            depth_min = depth.min()
            depth_max = depth.max()
            return ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        depth_original_vis = normalize_depth(disp_original_np)
        depth_augmented_vis = normalize_depth(disp_augmented_np)
        
        # Colormap 적용 (viridis)
        depth_original_colored = cv2.applyColorMap(depth_original_vis, cv2.COLORMAP_VIRIDIS)
        depth_augmented_colored = cv2.applyColorMap(depth_augmented_vis, cv2.COLORMAP_VIRIDIS)
        
        # RGB로 변환
        depth_original_colored = cv2.cvtColor(depth_original_colored, cv2.COLOR_BGR2RGB)
        depth_augmented_colored = cv2.cvtColor(depth_augmented_colored, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        depth_original_pil = pil.fromarray(depth_original_colored)
        depth_augmented_pil = pil.fromarray(depth_augmented_colored)
        
        # 개별 이미지 저장
        original_resized.save("test_1_original.png")
        augmented_img.save("test_2_augmented.png")
        depth_original_pil.save("test_3_original_depth.png")
        depth_augmented_pil.save("test_4_augmented_depth.png")
        
        print(f"✅ Saved individual images:")
        print(f"   - test_1_original.png (원본)")
        print(f"   - test_2_augmented.png (변환된 이미지)")
        print(f"   - test_3_original_depth.png (원본 추론)")
        print(f"   - test_4_augmented_depth.png (변환된 이미지 추론)")
        
        # 4장을 2x2 그리드로 합성
        grid_width = feed_width * 2
        grid_height = feed_height * 2
        grid_img = pil.new('RGB', (grid_width, grid_height))
        
        grid_img.paste(original_resized, (0, 0))
        grid_img.paste(augmented_img, (feed_width, 0))
        grid_img.paste(depth_original_pil, (0, feed_height))
        grid_img.paste(depth_augmented_pil, (feed_width, feed_height))
        
        grid_img.save("test_combined_grid.png")
        print(f"✅ Saved combined grid: test_combined_grid.png")
        print(f"   Layout: [원본 | 변환] / [원본추론 | 변환추론]")
        
    elif not os.path.exists(test_image_path):
        print(f"\n⚠️ test.png not found. Skipping visualization.")
    elif not best['augs']:
        print(f"\n⚠️ Best augmentation is 'Original' (no augmentation). Skipping visualization.")

    print("\n" + "=" * 80); print("🎉 ADVANCED AUGMENTATION SEARCH COMPLETE! 🎉"); print("=" * 80)

if __name__ == '__main__':
    main()
