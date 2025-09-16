# from __future__ import absolute_import, division, print_function
# import os
# import sys
# from pathlib import Path
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from layers import disp_to_depth
# from utils import readlines
# from options import LiteMonoOptions
# import datasets
# import time
# from thop import clever_format
# from thop import profile
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import importlib.util


# cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

# splits_dir = os.path.join(os.path.dirname(__file__), "splits")


# def load_model_class(file_path, class_name):
#     """Dynamically load a class from a Python file"""
#     spec = importlib.util.spec_from_file_location("module", file_path)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules["module"] = module
#     spec.loader.exec_module(module)
#     return getattr(module, class_name)


# def profile_once(encoder, decoder, x):
#     x_e = x[0, :, :, :].unsqueeze(0)
#     x_d = encoder(x_e)
#     flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
#     flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

#     flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
#     flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
#     flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

#     return flops, params, flops_e, params_e, flops_d, params_d


# def time_sync():
#     # PyTorch-accurate time
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     return time.time()


# def compute_errors(gt, pred):
#     """Computation of error metrics between predicted and ground truth depths
#     """
#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25     ).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()

#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())

#     rmse_log = (np.log(gt) - np.log(pred)) ** 2
#     rmse_log = np.sqrt(rmse_log.mean())

#     abs_rel = np.mean(np.abs(gt - pred) / gt)

#     sq_rel = np.mean(((gt - pred) ** 2) / gt)

#     return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# def batch_post_process_disparity(l_disp, r_disp):
#     """Apply the disparity post-processing method as introduced in Monodepthv1
#     """
#     _, h, w = l_disp.shape
#     m_disp = 0.5 * (l_disp + r_disp)
#     l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
#     l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
#     r_mask = l_mask[:, :, ::-1]
#     return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


# def disparity_to_colormap(disparity, vmin=None, vmax=None):
#     """Convert disparity map to colormap for visualization using matplotlib's standard approach"""
#     import matplotlib as mpl
#     import matplotlib.cm as cm
#     import numpy as np
    
#     # Set vmin if not provided
#     if vmin is None:
#         vmin = disparity.min()
    
#     # Use 95th percentile for vmax to avoid outliers (more robust)
#     if vmax is None:
#         vmax = np.percentile(disparity, 95)
    
#     # Use matplotlib's Normalize class for proper normalization
#     normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
#     # Create ScalarMappable with magma colormap (same as second method)
#     # For disparity: high disparity = close = bright colors
#     mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    
#     # Apply colormap and convert to RGB
#     colormapped_im = mapper.to_rgba(disparity)[:, :, :3]  # Remove alpha channel
#     colormap_rgb = (colormapped_im * 255).astype(np.uint8)
    
#     return colormap_rgb


# def gt_to_grayscale(gt_depth, vmin=None, vmax=None):
#     """Convert ground truth depth to grayscale for visualization"""
#     import cv2
#     import numpy as np
    
#     # Set range if not provided
#     if vmin is None:
#         vmin = gt_depth.min()
#     if vmax is None:
#         vmax = gt_depth.max()
    
#     # Normalize to 0-255 range
#     normalized = cv2.normalize(gt_depth, None, 0, 255, cv2.NORM_MINMAX)
#     grayscale = normalized.astype(np.uint8)
    
#     # Convert to RGB format (3 channels) for consistency with other images
#     gt_rgb = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    
#     return gt_rgb


# def compute_error_map(gt_depth, pred_depth, mask=None, error_type='abs_rel'):
#     """Compute error map between ground truth and predicted depth"""
#     if mask is not None:
#         # Apply mask to both depths
#         gt_masked = gt_depth.copy()
#         pred_masked = pred_depth.copy()
#         gt_masked[~mask] = 0
#         pred_masked[~mask] = 0
#     else:
#         gt_masked = gt_depth
#         pred_masked = pred_depth
    
#     # Avoid division by zero
#     gt_safe = np.where(gt_masked > 1e-6, gt_masked, 1e-6)
#     pred_safe = np.where(pred_masked > 1e-6, pred_masked, 1e-6)
    
#     if error_type == 'abs_rel':
#         error_map = np.abs(gt_safe - pred_safe) / gt_safe
#     elif error_type == 'sq_rel':
#         error_map = ((gt_safe - pred_safe) ** 2) / gt_safe
#     elif error_type == 'rmse':
#         error_map = np.abs(gt_safe - pred_safe)
#     elif error_type == 'rmse_log':
#         error_map = np.abs(np.log(gt_safe) - np.log(pred_safe))
#     else:
#         raise ValueError(f"Unknown error type: {error_type}")
    
#     # Set masked areas to 0
#     if mask is not None:
#         error_map[~mask] = 0
    
#     return error_map


# def error_map_to_colormap(error_map, error_type='abs_rel', mask=None):
#     """Convert error map to colormap for visualization with intuitive colors"""
#     if mask is not None:
#         # Only consider masked areas for normalization
#         masked_errors = error_map[mask]
#         if len(masked_errors) > 0:
#             vmin = masked_errors.min()
#             vmax = np.percentile(masked_errors, 95)  # Use 95th percentile to avoid outliers
#         else:
#             vmin, vmax = 0, 1
#     else:
#         vmin = error_map.min()
#         vmax = np.percentile(error_map.flatten(), 95)
    
#     # Normalize error map to 0-1 range
#     normalized_error = np.clip((error_map - vmin) / (vmax - vmin), 0, 1)
    
#     # Apply coolwarm colormap (Blue=good, Red=bad)
#     # Ensure no white areas by clipping to valid range
#     normalized_error = np.clip(normalized_error, 0.001, 0.999)  # Avoid pure white
#     colormap = cm.coolwarm(normalized_error)
#     colormap_rgb = (colormap[:, :, :3] * 255).astype(np.uint8)
    
#     # Set masked areas to black if mask is provided
#     if mask is not None:
#         colormap_rgb[~mask] = [0, 0, 0]
    
#     return colormap_rgb


# def add_text_to_image(img, text, position='bottom', font_scale=0.8, thickness=2, margin=10):
#     """Add text to image with background"""
#     font = cv2.FONT_HERSHEY_SIMPLEX
    
#     # Get text size
#     text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
#     text_width, text_height = text_size
    
#     # Create space for text
#     if position == 'bottom':
#         text_bg_height = text_height + 2 * margin
#         # Create new image with space for text
#         new_img = np.ones((img.shape[0] + text_bg_height, img.shape[1], img.shape[2]), dtype=img.dtype) * 255
#         new_img[:img.shape[0], :, :] = img
        
#         # Calculate text position (center horizontally)
#         text_x = (img.shape[1] - text_width) // 2
#         text_y = img.shape[0] + margin + text_height
        
#         # Add text
#         cv2.putText(new_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
#     return new_img


# def save_comparison_images(original_images, pred_disps, gt_depths, errors_per_image, 
#                           filenames, save_dir, opt):
#     """Save best and worst images for each metric"""
    
#     os.makedirs(save_dir, exist_ok=True)
    
#     metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log']
    
#     # Convert errors to numpy array for easier indexing
#     errors_array = np.array(errors_per_image)
    
#     for metric_idx, metric_name in enumerate(metrics):
#         metric_errors = errors_array[:, metric_idx]
        
#         # Find best (lowest error) and worst (highest error) indices
#         best_idx = np.argmin(metric_errors)
#         worst_idx = np.argmax(metric_errors)
        
#         for idx, label in [(best_idx, 'best'), (worst_idx, 'worst')]:
#             # Get original image
#             orig_img = original_images[idx]
#             if orig_img.shape[0] == 3:  # If CHW format, convert to HWC
#                 orig_img = orig_img.transpose(1, 2, 0)
            
#             # Convert from [0,1] to [0,255] if needed
#             if orig_img.max() <= 1.0:
#                 orig_img = (orig_img * 255).astype(np.uint8)
            
#             # Get predicted disparity and convert to colormap
#             pred_disp = pred_disps[idx]
#             pred_disp_colored = disparity_to_colormap(pred_disp)
            
#             # Get ground truth depth and convert to grayscale
#             gt_depth = gt_depths[idx]
#             gt_depth_colored = gt_to_grayscale(gt_depth)
            
#             # For error map computation, we need depth from disparity
#             # Resize disparity to match GT size first
#             pred_disp_resized = cv2.resize(pred_disp, (gt_depth.shape[1], gt_depth.shape[0]))
#             pred_depth = 1 / pred_disp_resized
            
#             # Create mask for evaluation (same as in main evaluation loop)
#             MIN_DEPTH = 1e-3
#             MAX_DEPTH = 80
#             gt_height, gt_width = gt_depth.shape[:2]
            
#             if opt.eval_split == "eigen":
#                 mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
#                 crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
#                                 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
#                 crop_mask = np.zeros(mask.shape)
#                 crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
#                 mask = np.logical_and(mask, crop_mask)
#             else:
#                 mask = gt_depth > 0
            
#             # Compute error map
#             error_map = compute_error_map(gt_depth, pred_depth, mask, metric_name)
#             error_map_colored = error_map_to_colormap(error_map, metric_name, mask)
            
#             # Resize images to match if necessary
#             h, w = orig_img.shape[:2]
#             if pred_disp_colored.shape[:2] != (h, w):
#                 pred_disp_colored = cv2.resize(pred_disp_colored, (w, h))
#             if gt_depth_colored.shape[:2] != (h, w):
#                 gt_depth_colored = cv2.resize(gt_depth_colored, (w, h))
#             if error_map_colored.shape[:2] != (h, w):
#                 error_map_colored = cv2.resize(error_map_colored, (w, h))
            
#             # Add titles to each image
#             orig_img_titled = add_text_to_image(orig_img, "Original Image")
#             pred_img_titled = add_text_to_image(pred_disp_colored, "Predicted Disparity")
#             gt_img_titled = add_text_to_image(gt_depth_colored, "Ground Truth Depth")
#             error_img_titled = add_text_to_image(error_map_colored, f"{metric_name.upper()} Error Map")
            
#             # Create 2x2 layout: top row (original, disparity), bottom row (GT, error)
#             top_row = np.hstack([orig_img_titled, pred_img_titled])
#             bottom_row = np.hstack([gt_img_titled, error_img_titled])
#             combined_img = np.vstack([top_row, bottom_row])
            
#             # Add main title to the combined image
#             error_value = metric_errors[idx]
#             main_title = f"{metric_name.upper()} {label.upper()} - Error: {error_value:.4f}"
#             combined_img_titled = add_text_to_image(combined_img, main_title, font_scale=1.2, thickness=3)
            
#             # Save image
#             filename = filenames[idx].split('/')[-1].replace('.jpg', '').replace('.png', '')
#             save_path = os.path.join(save_dir, f"{metric_name}_{label}_{filename}_error_{metric_errors[idx]:.4f}.jpg")
            
#             cv2.imwrite(save_path, cv2.cvtColor(combined_img_titled, cv2.COLOR_RGB2BGR))
#             print(f"Saved {metric_name} {label} image: {save_path}")



# def evaluate(opt, weight_path=None, model_name=None):
#     """Evaluates a pretrained model using a specified test set
#     """
#     MIN_DEPTH = 1e-3
#     MAX_DEPTH = 80

#     #add
#     if weight_path is not None:
#         opt.load_weights_folder = weight_path
        
#     if opt.ext_disp_to_eval is None:

#         opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

#         assert os.path.isdir(opt.load_weights_folder), \
#             "Cannot find a folder at {}".format(opt.load_weights_folder)

#         print("-> Loading weights from {}".format(opt.load_weights_folder))

#         filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
#         encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
#         decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

#         encoder_dict = torch.load(encoder_path)
#         decoder_dict = torch.load(decoder_path)

#         dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
#                                         encoder_dict['height'], encoder_dict['width'],
#                                         [0], 4, is_train=False)
#         dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
#                                 pin_memory=True, drop_last=False)

#         # Load encoder and decoder classes from specific files
#         if model_name is None:
#             # Extract model name from load_weights_folder if not provided
#             path = Path(opt.load_weights_folder)
#             model_name = path.name
        
#         encoder_file_path = f"experiments/logs/{model_name}/{model_name}_encoder.py"
#         decoder_file_path = f"experiments/logs/{model_name}/{model_name}_decoder.py"
        
#         print(f"-> Loading LiteMono class from {encoder_file_path}")
#         print(f"-> Loading DepthDecoder class from {decoder_file_path}")
        
#         # Dynamically load the classes
#         LiteMono = load_model_class(encoder_file_path, "LiteMono")
#         DepthDecoder = load_model_class(decoder_file_path, "DepthDecoder")
        
#         # Create instances using the dynamically loaded classes
#         encoder = LiteMono(model=opt.model,
#                           height=encoder_dict['height'],
#                           width=encoder_dict['width'])
#         depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
        
#         model_dict = encoder.state_dict()
#         depth_model_dict = depth_decoder.state_dict()
#         encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
#         depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

#         encoder.cuda()
#         encoder.eval()
#         depth_decoder.cuda()
#         depth_decoder.eval()

#         pred_disps = []
#         original_images = []  # Store original images

#         print("-> Computing predictions with size {}x{}".format(
#             encoder_dict['width'], encoder_dict['height']))

#         with torch.no_grad():
#             for data in dataloader:
#                 input_color = data[("color", 0, 0)].cuda()
                
#                 # Store original images
#                 original_images.extend(input_color.cpu().numpy())

#                 if opt.post_process:
#                     # Post-processed results require each image to have two forward passes
#                     input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

#                 flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
#                 t1 = time_sync()
#                 output = depth_decoder(encoder(input_color))
#                 t2 = time_sync()

#                 pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
#                 pred_disp = pred_disp.cpu()[:, 0].numpy()

#                 if opt.post_process:
#                     N = pred_disp.shape[0] // 2
#                     pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
#                     # Also trim original_images to match
#                     original_images = original_images[:len(original_images)//2]

#                 pred_disps.append(pred_disp)

#         pred_disps = np.concatenate(pred_disps)

#     else:
#         # Load predictions from file
#         print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
#         pred_disps = np.load(opt.ext_disp_to_eval)
#         original_images = []  # Empty list when loading from file

#         if opt.eval_eigen_to_benchmark:
#             eigen_to_benchmark_ids = np.load(
#                 os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

#             pred_disps = pred_disps[eigen_to_benchmark_ids]

#     if opt.save_pred_disps:
#         output_path = os.path.join(
#             opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
#         print("-> Saving predicted disparities to ", output_path)
#         np.save(output_path, pred_disps)

#     if opt.no_eval:
#         print("-> Evaluation disabled. Done.")
#         quit()

#     gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
#     gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

#     print("-> Evaluating")
#     print("   Mono evaluation - using median scaling")

#     errors = []
#     ratios = []
#     pred_depths_processed = []  # Store processed prediction depths
#     gt_depths_processed = []    # Store processed ground truth depths
#     errors_per_image = []       # Store individual image errors

#     for i in range(pred_disps.shape[0]):
#         gt_depth = gt_depths[i]
#         gt_height, gt_width = gt_depth.shape[:2]

#         pred_disp = pred_disps[i]
#         pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
#         pred_depth = 1 / pred_disp

#         if opt.eval_split == "eigen":
#             mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

#             crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
#                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
#             crop_mask = np.zeros(mask.shape)
#             crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
#             mask = np.logical_and(mask, crop_mask)

#         else:
#             mask = gt_depth > 0

#         pred_depth_masked = pred_depth[mask]
#         gt_depth_masked = gt_depth[mask]

#         pred_depth_masked *= opt.pred_depth_scale_factor
#         if not opt.disable_median_scaling:
#             ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
#             ratios.append(ratio)
#             pred_depth_masked *= ratio
#             # Apply ratio to full image for visualization
#             pred_depth *= ratio

#         pred_depth_masked[pred_depth_masked < MIN_DEPTH] = MIN_DEPTH
#         pred_depth_masked[pred_depth_masked > MAX_DEPTH] = MAX_DEPTH
        
#         # Compute errors for this image
#         image_errors = compute_errors(gt_depth_masked, pred_depth_masked)
#         errors.append(image_errors)
#         errors_per_image.append(image_errors)
        
#         # Store processed depths for visualization (full images, not masked)
#         pred_depths_processed.append(pred_depth)
#         gt_depths_processed.append(gt_depth)

#     if not opt.disable_median_scaling:
#         ratios = np.array(ratios)
#         med = np.median(ratios)
#         print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

#     mean_errors = np.array(errors).mean(0)

#     if weight_path is not None:
#         path = Path(opt.load_weights_folder)
#         filename = path.name
#         print(filename + "\n")
    
#     print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
#     print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
#     print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops, params, flops_e, params_e, flops_d, params_d))
    
#     # Save best and worst images for each metric
#     if original_images:  # Only save if we have original images
#         save_dir = os.path.join(opt.load_weights_folder, "comparison_images")
#         # save_dir = rf"C:\Users\wodud\OneDrive\Desktop\도로주행 데이터\{model_name}\comparison_images"
#         save_comparison_images(original_images, pred_disps, gt_depths_processed, 
#                              errors_per_image, filenames, save_dir, opt)
    
#     print("\n-> Done!")

#     return mean_errors


# if __name__ == "__main__":
#     options = LiteMonoOptions()   
#     name = "v4_2_1"
#     path = f"experiments/logs/{name}/models/weights_19"
#     evaluate(options.parse(),path,name)
    
    
    
from __future__ import absolute_import, division, print_function
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import datasets
import time
from thop import clever_format
from thop import profile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import importlib.util

# Matplotlib Agg backend 설정 (GUI 없는 환경에서도 실행 가능하도록)
import matplotlib
matplotlib.use('Agg')


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def load_model_class(file_path, class_name):
    """Dynamically load a class from a Python file"""
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def disparity_to_colormap(disparity, vmin=None, vmax=None):
    """Convert disparity map to colormap for visualization"""
    if vmin is None:
        vmin = disparity.min()
    if vmax is None:
        vmax = np.percentile(disparity, 95)
    
    normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disparity)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


def gt_to_grayscale(gt_depth, vmin=None, vmax=None):
    """Convert ground truth depth to grayscale for visualization"""
    if vmin is None:
        vmin = gt_depth.min()
    if vmax is None:
        vmax = gt_depth.max()
    
    normalized = cv2.normalize(gt_depth, None, 0, 255, cv2.NORM_MINMAX)
    grayscale = normalized.astype(np.uint8)
    gt_rgb = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    return gt_rgb


def compute_error_map(gt_depth, pred_depth, mask=None, error_type='abs_rel'):
    """Compute error map between ground truth and predicted depth"""
    gt_masked = gt_depth.copy()
    pred_masked = pred_depth.copy()
    if mask is not None:
        gt_masked[~mask] = 0
        pred_masked[~mask] = 0
    
    gt_safe = np.where(gt_masked > 1e-6, gt_masked, 1e-6)
    pred_safe = np.where(pred_masked > 1e-6, pred_masked, 1e-6)
    
    if error_type == 'abs_rel':
        error_map = np.abs(gt_safe - pred_safe) / gt_safe
    elif error_type == 'sq_rel':
        error_map = ((gt_safe - pred_safe) ** 2) / gt_safe
    elif error_type == 'rmse':
        error_map = np.abs(gt_safe - pred_safe)
    elif error_type == 'rmse_log':
        error_map = np.abs(np.log(gt_safe) - np.log(pred_safe))
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    
    if mask is not None:
        error_map[~mask] = 0
    return error_map


def error_map_to_colormap(error_map, mask=None):
    """Convert error map to colormap for visualization"""
    if mask is not None and mask.any():
        masked_errors = error_map[mask]
        vmin = masked_errors.min()
        vmax = np.percentile(masked_errors, 95)
    else:
        vmin = error_map.min()
        vmax = np.percentile(error_map.flatten(), 95)

    if vmax <= vmin:
        vmax = vmin + 1e-6

    normalized_error = np.clip((error_map - vmin) / (vmax - vmin), 0, 1)
    colormap = cm.coolwarm(normalized_error)
    colormap_rgb = (colormap[:, :, :3] * 255).astype(np.uint8)
    
    if mask is not None:
        colormap_rgb[~mask] = [0, 0, 0] # Masked areas are black
    
    return colormap_rgb


def add_text_to_image(img, text, position='bottom', font_scale=0.8, thickness=2, margin=10):
    """Add text to an image with a background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    
    img_h, img_w = img.shape[:2]
    
    if position == 'bottom':
        bg_height = text_height + 2 * margin
        new_img = np.full((img_h + bg_height, img_w, 3), 255, dtype=np.uint8)
        new_img[:img_h, :, :] = img
        
        text_x = (img_w - text_width) // 2
        text_y = img_h + margin + text_height
        
        cv2.putText(new_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return new_img
    else: # top position
        bg_height = text_height + 2 * margin
        new_img = np.full((img_h + bg_height, img_w, 3), 255, dtype=np.uint8)
        new_img[bg_height:, :, :] = img
        
        text_x = (img_w - text_width) // 2
        text_y = margin + text_height

        cv2.putText(new_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return new_img


def create_comparison_image(orig_img, pred_disp, gt_depth, metric_name, error_value, opt):
    """Helper function to create a 2x2 comparison grid for a single image"""
    
    # Prepare original image
    if orig_img.shape[0] == 3: # CHW to HWC
        orig_img = orig_img.transpose(1, 2, 0)
    if orig_img.max() <= 1.0:
        orig_img = (orig_img * 255).astype(np.uint8)

    # Prepare disparity and ground truth
    pred_disp_colored = disparity_to_colormap(pred_disp)
    gt_depth_colored = gt_to_grayscale(gt_depth)

    # Prepare error map
    pred_disp_resized = cv2.resize(pred_disp, (gt_depth.shape[1], gt_depth.shape[0]))
    pred_depth = 1 / pred_disp_resized
    
    # Create mask
    MIN_DEPTH, MAX_DEPTH = 1e-3, 80
    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
    if opt.eval_split == "eigen":
        gt_height, gt_width = gt_depth.shape[:2]
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
    
    error_map = compute_error_map(gt_depth, pred_depth, mask, metric_name)
    error_map_colored = error_map_to_colormap(error_map, mask)
    
    # Resize all components to match original image size for stacking
    h, w = orig_img.shape[:2]
    pred_disp_colored = cv2.resize(pred_disp_colored, (w, h))
    gt_depth_colored = cv2.resize(gt_depth_colored, (w, h))
    error_map_colored = cv2.resize(error_map_colored, (w, h))

    # Add titles to each component
    orig_img_titled = add_text_to_image(orig_img, "Original Image")
    pred_img_titled = add_text_to_image(pred_disp_colored, "Predicted Disparity")
    gt_img_titled = add_text_to_image(gt_depth_colored, "Ground Truth Depth")
    error_img_titled = add_text_to_image(error_map_colored, f"{metric_name.upper()} Error Map")
    
    # Combine into a 2x2 grid
    top_row = np.hstack([orig_img_titled, pred_img_titled])
    bottom_row = np.hstack([gt_img_titled, error_img_titled])
    combined_img = np.vstack([top_row, bottom_row])
    
    # Add main title
    main_title = f"Error: {error_value:.4f}"
    combined_img_titled = add_text_to_image(combined_img, main_title, position='top', font_scale=1.2, thickness=3)
    
    return combined_img_titled


def save_comparison_images(original_images, pred_disps, gt_depths, errors_per_image,
                           filenames, save_dir, opt, num_to_save=5):
    """Save best and worst N images for each metric"""
    os.makedirs(save_dir, exist_ok=True)
    metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log']
    errors_array = np.array(errors_per_image)
    
    for metric_idx, metric_name in enumerate(metrics):
        metric_errors = errors_array[:, metric_idx]
        sorted_indices = np.argsort(metric_errors)
        
        best_indices = sorted_indices[:num_to_save]
        worst_indices = sorted_indices[-num_to_save:]
        
        print(f"\n-> Saving comparison images for metric: {metric_name.upper()}")
        
        for rank, idx in enumerate(best_indices):
            label = f'best_{rank+1}'
            img_to_save = create_comparison_image(original_images[idx], pred_disps[idx], gt_depths[idx], metric_name, metric_errors[idx], opt)
            filename_stem = Path(filenames[idx]).stem
            save_path = os.path.join(save_dir, f"{metric_name}_{label}_{filename_stem}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
            print(f"Saved {label} image: {save_path}")

        for rank, idx in enumerate(worst_indices[::-1]): # Reverse for worst_1, worst_2...
            label = f'worst_{rank+1}'
            img_to_save = create_comparison_image(original_images[idx], pred_disps[idx], gt_depths[idx], metric_name, metric_errors[idx], opt)
            filename_stem = Path(filenames[idx]).stem
            save_path = os.path.join(save_dir, f"{metric_name}_{label}_{filename_stem}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
            print(f"Saved {label} image: {save_path}")


def evaluate(opt, weight_path=None, model_name=None):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    if weight_path is not None:
        opt.load_weights_folder = weight_path
        
    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        if model_name is None:
            path = Path(opt.load_weights_folder)
            model_name = path.name
        
        encoder_file_path = f"experiments/logs/{model_name}/{model_name}_encoder.py"
        decoder_file_path = f"experiments/logs/{model_name}/{model_name}_decoder.py"
        
        LiteMono = load_model_class(encoder_file_path, "LiteMono")
        DepthDecoder = load_model_class(decoder_file_path, "DepthDecoder")
        
        encoder = LiteMono(model=opt.model, height=encoder_dict['height'], width=encoder_dict['width'])
        depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
        
        model_dict = encoder.state_dict()
        depth_model_dict = depth_decoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []
        original_images = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                
                if opt.post_process:
                    original_images.extend(input_color.cpu().numpy())
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                else:
                    original_images.extend(input_color.cpu().numpy())

                flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                    original_images = original_images[:N] # Trim original images to match post-processed disps

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)
        original_images = []

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("\n-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    errors_per_image = []

    # --- 추가된 시각화를 위한 변수 초기화 ---
    h, w = encoder_dict['height'], encoder_dict['width']
    total_error_map = np.zeros((h, w), dtype=np.float32)
    gt_depth_points = []
    error_points = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        if opt.eval_split == "eigen":
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        
        pred_depth_masked = pred_depth[mask]
        gt_depth_masked = gt_depth[mask]

        pred_depth_masked *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
            ratios.append(ratio)
            pred_depth_masked *= ratio
        
        pred_depth_masked[pred_depth_masked < MIN_DEPTH] = MIN_DEPTH
        pred_depth_masked[pred_depth_masked > MAX_DEPTH] = MAX_DEPTH
        
        image_errors = compute_errors(gt_depth_masked, pred_depth_masked)
        errors.append(image_errors)
        errors_per_image.append(image_errors)

        # --- 종합 에러 맵 및 산점도 데이터 누적 ---
        scaled_pred_depth = pred_depth * (ratio if not opt.disable_median_scaling else 1.0)
        
        # 1. 종합 에러 맵
        error_map = compute_error_map(gt_depth, scaled_pred_depth, mask, error_type='abs_rel')
        resized_error_map = cv2.resize(error_map, (w, h))
        total_error_map += resized_error_map

        # 2. 산점도 데이터
        abs_error = np.abs(gt_depth_masked - pred_depth_masked)
        if len(gt_depth_masked) > 1000: # 샘플링
             sample_indices = np.random.choice(len(gt_depth_masked), 1000, replace=False)
             gt_depth_points.extend(gt_depth_masked[sample_indices])
             error_points.extend(abs_error[sample_indices])
        else:
             gt_depth_points.extend(gt_depth_masked)
             error_points.extend(abs_error)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops, params, flops_e, params_e, flops_d, params_d))
    
    # --- 추가된 시각화 결과물 저장 ---
    save_dir_base = opt.load_weights_folder

    # 1. 종합 에러 맵 저장
    mean_error_map = total_error_map / len(pred_disps)
    agg_error_map_colored = error_map_to_colormap(mean_error_map)
    agg_save_path = os.path.join(save_dir_base, "aggregated_error_map.jpg")
    cv2.imwrite(agg_save_path, cv2.cvtColor(agg_error_map_colored, cv2.COLOR_RGB2BGR))
    print(f"\n-> Saved aggregated error map to {agg_save_path}")

    # 2. 오차 vs. 깊이 산점도 저장
    plt.figure(figsize=(10, 6))
    plt.scatter(gt_depth_points, error_points, s=1, alpha=0.1)
    plt.xlabel("Ground Truth Depth (m)")
    plt.ylabel("Absolute Error (m)")
    plt.title("Error vs. Depth")
    plt.grid(True)
    plt.xlim(0, MAX_DEPTH)
    plt.ylim(0, max(error_points) if error_points else 10)
    scatter_save_path = os.path.join(save_dir_base, "error_vs_depth_scatter.png")
    plt.savefig(scatter_save_path)
    plt.close() # Figure 닫기
    print(f"-> Saved error vs. depth scatter plot to {scatter_save_path}")
    
    # 3. 상위/하위 N개 및 무작위 샘플 이미지 저장
    if original_images:
        # 상위/하위 N개 저장
        comp_save_dir = os.path.join(save_dir_base, "comparison_images_best_worst")
        save_comparison_images(original_images, pred_disps, gt_depths, errors_per_image, 
                               filenames, comp_save_dir, opt, num_to_save=5)
        
        # 무작위 샘플 N개 저장
        rand_save_dir = os.path.join(save_dir_base, "comparison_images_random")
        os.makedirs(rand_save_dir, exist_ok=True)
        num_random_samples = 5
        random_indices = np.random.choice(len(original_images), num_random_samples, replace=False)
        
        print(f"\n-> Saving {num_random_samples} random samples...")
        for idx in random_indices:
            # 여기서는 abs_rel 에러를 기준으로 제목을 만듦
            error_value = errors_per_image[idx][0] 
            img_to_save = create_comparison_image(original_images[idx], pred_disps[idx], gt_depths[idx], 'abs_rel', error_value, opt)
            filename_stem = Path(filenames[idx]).stem
            save_path = os.path.join(rand_save_dir, f"random_sample_{filename_stem}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
            print(f"Saved random sample: {save_path}")

    print("\n-> Done!")
    return mean_errors


if __name__ == "__main__":
    options = LiteMonoOptions()
    opts = options.parse()
    # 모델 이름과 가중치 경로를 직접 지정
    model_name = "v4_2_1_1" 
    weights_path = f"experiments/logs/{model_name}/models/weights_19"
    evaluate(opts, weights_path, model_name)