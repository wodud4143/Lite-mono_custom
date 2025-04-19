from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
import cv2
import heapq
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str, nargs='+',
                        help='path to a test image or folder of images',
                        default= direct()
                        )# required=True

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use',
                        default=r'C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\Depth_wise\models\weights_49'
                        )

    parser.add_argument('--test',
                        action='store_true',
                        help='if set, read images from a .txt file',
                        # default=r'splits\eigen\test_files.txt'
                        )

    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="lite-mono", #lite-mono
                        choices=[
                            "lite-mono",
                            "lite-mono-small",
                            "lite-mono-tiny",
                            "lite-mono-8m"])

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.load_weights_folder is not None, \
        "You must specify the --load_weights_folder parameter"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.LiteMono(model=args.model,
                                    height=feed_height,
                                    width=feed_width)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()
    
    '''
    # 하위 디렉토리 많을때
    '''
    # for image_folder in args.image_path:
    image_folder = r"C:\Users\wodud\OneDrive\Desktop\sample\test2_DC"
    if image_folder :
        # FINDING INPUT IMAGES
        if os.path.isfile(image_folder) and not args.test:
            # Only testing on a single image
            paths = [image_folder]
            output_directory = os.path.dirname(image_folder)
        elif os.path.isfile(image_folder) and args.test:
            gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
            gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
            output_directory = "output1"

            side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
            # reading images from .txt file
            paths = []
            with open(image_folder) as f:
                filenames = f.readlines()
                for i in range(len(filenames)):
                    filename = filenames[i]
                    line = filename.split()
                    folder = line[0]
                    if len(line) == 3:
                        frame_index = int(line[1])
                        side = line[2]

                    f_str = "{:010d}{}".format(frame_index, '.png') #jpg
                    image_path = os.path.join(
                        'kitti_data',
                        folder,
                        "image_0{}/data".format(side_map[side]),
                        f_str)
                    paths.append(image_path)

        elif os.path.isdir(image_folder):
            # Searching folder for images
            paths = glob.glob(os.path.join(image_folder, '*.{}'.format(args.ext)))
            output_directory = image_folder
        else:
            raise Exception("Can not find args.image_path: {}".format(image_folder))

        print("-> Predicting on {:d} test images".format(len(paths)))

        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            for idx, image_path in enumerate(paths):

                if image_path.endswith("_disp.jpg"):
                    # don't try to predict disparity for a disparity image!
                    continue
                
                

                if not os.path.exists(image_path):
                    print(f"Warning: File not found - {image_path}")
                    continue  
                            
                
                # Load image and preprocess
                input_image = pil.open(image_path).convert('RGB')
                original_width, original_height = input_image.size
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # PREDICTION
                input_image = input_image.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]

                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                # Saving numpy file
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                # output_name = os.path.splitext(image_path)[0].split('/')[-1]
                scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                
                # # # 원본이미지 같이 저장
                # original_image = pil.open(image_path)
                # original_image.save(os.path.join(output_directory, "{}.png".format(output_name)))
                
                

                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)

                name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
                im.save(name_dest_im)

                print("   Processed {:d} of {:d} images - saved predictions to:".format(
                    idx + 1, len(paths)))
                print("   - {}".format(name_dest_im))
                print("   - {}".format(name_dest_npy))


        print('-> Done!')

    # # FINDING INPUT IMAGES
    # if os.path.isfile(args.image_path) and not args.test:
    #     # Only testing on a single image
    #     paths = [args.image_path]
    #     output_directory = os.path.dirname(args.image_path)
    # elif os.path.isfile(args.image_path) and args.test:
    #     gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
    #     gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    #     output_directory = "output1"

    #     side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    #     # reading images from .txt file
    #     paths = []
    #     with open(args.image_path) as f:
    #         filenames = f.readlines()
    #         for i in range(len(filenames)):
    #             filename = filenames[i]
    #             line = filename.split()
    #             folder = line[0]
    #             if len(line) == 3:
    #                 frame_index = int(line[1])
    #                 side = line[2]

    #             f_str = "{:010d}{}".format(frame_index, '.png') #jpg
    #             image_path = os.path.join(
    #                 'kitti_data',
    #                 folder,
    #                 "image_0{}/data".format(side_map[side]),
    #                 f_str)
    #             paths.append(image_path)

    # elif os.path.isdir(args.image_path):
    #     # Searching folder for images
    #     paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    #     output_directory = args.image_path
    # else:
    #     raise Exception("Can not find args.image_path: {}".format(args.image_path))

    # print("-> Predicting on {:d} test images".format(len(paths)))

    # # PREDICTING ON EACH IMAGE IN TURN
    # with torch.no_grad():
    #     for idx, image_path in enumerate(paths):

    #         if image_path.endswith("_disp.jpg"):
    #             # don't try to predict disparity for a disparity image!
    #             continue
            
            

    #         if not os.path.exists(image_path):
    #             print(f"Warning: File not found - {image_path}")
    #             continue  
                        
            
    #         # Load image and preprocess
    #         input_image = pil.open(image_path).convert('RGB')
    #         original_width, original_height = input_image.size
    #         input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    #         input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    #         # PREDICTION
    #         input_image = input_image.to(device)
    #         features = encoder(input_image)
    #         outputs = depth_decoder(features)

    #         disp = outputs[("disp", 0)]

    #         disp_resized = torch.nn.functional.interpolate(
    #             disp, (original_height, original_width), mode="bilinear", align_corners=False)

    #         # Saving numpy file
    #         output_name = os.path.splitext(os.path.basename(image_path))[0]
    #         # output_name = os.path.splitext(image_path)[0].split('/')[-1]
    #         scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            
    #         # # # 원본이미지 같이 저장
    #         # original_image = pil.open(image_path)
    #         # original_image.save(os.path.join(output_directory, "{}.png".format(output_name)))
            
            

    #         name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
    #         np.save(name_dest_npy, scaled_disp.cpu().numpy())

    #         # Saving colormapped depth image
    #         disp_resized_np = disp_resized.squeeze().cpu().numpy()
    #         vmax = np.percentile(disp_resized_np, 95)
    #         normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    #         mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    #         colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    #         im = pil.fromarray(colormapped_im)

    #         name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
    #         im.save(name_dest_im)

    #         print("   Processed {:d} of {:d} images - saved predictions to:".format(
    #             idx + 1, len(paths)))
    #         print("   - {}".format(name_dest_im))
    #         print("   - {}".format(name_dest_npy))


    # print('-> Done!')
    

def direct():

    directory = r"C:\Users\wodud\OneDrive\Desktop\Sample" 

    folders = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    print("디렉토리 내 폴더 목록:")
    for folder in folders:
        print(folder)
    
    return folders

# def direct():
#     gt_root = r"C:\Users\wodud\OneDrive\Desktop\data_depth_annotated"
#     rgb_root = r"C:\Users\wodud\OneDrive\Desktop\Develop\Lite-Mono\kitti_data"
#     target_cams = ["image_02", "image_03"]
#     image_dirs = set()

#     for split in ["train", "val"]:
#         split_path = os.path.join(gt_root, split)
#         if not os.path.exists(split_path):
#             continue

#         for drive in os.listdir(split_path):
#             drive_path = os.path.join(split_path, drive, "proj_depth", "groundtruth")
#             for cam in target_cams:
#                 gt_dir = os.path.join(drive_path, cam)
#                 if not os.path.exists(gt_dir):
#                     continue

#                 date = drive.split("_drive_")[0]
#                 img_dir = os.path.join(rgb_root, date, drive, cam, "data")
#                 if os.path.exists(img_dir):
#                     image_dirs.add(img_dir)

#     image_dirs = sorted(image_dirs)
#     print(f"[GT 기반 유효 디렉토리 수] {len(image_dirs)}개")
#     return image_dirs



if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
