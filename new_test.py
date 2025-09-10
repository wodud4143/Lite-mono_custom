from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import importlib.util

import torch
from torchvision import transforms, datasets

from layers import disp_to_depth, transformation_from_parameters
import cv2
import heapq
from PIL import ImageFile

import torch.nn.functional as F
from options import LiteMonoOptions
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str, nargs='+',
                        help='path to a test image or folder of images',
                        default= direct()
                        )# required=True

    parser.add_argument('--model_name', type=str,
                        help='model name for path construction',
                        default='v4.1'
                        )

    parser.add_argument('--test',
                        action='store_true',
                        help='if set, read images from a .txt file',
                        )

    parser.add_argument('--model', type=str,
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


def load_module_from_path(module_name, file_path):
    """동적으로 모듈을 로드하는 함수"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    # 모델명 설정
    modelname = args.model_name
    
    # 절대 경로 설정
    base_path = os.path.abspath("experiments/logs")
    model_base_path = os.path.join(base_path, modelname)
    
    # 가중치 파일 경로
    encoder_path = os.path.join(model_base_path, "models", "weights_19", "encoder.pth")
    decoder_path = os.path.join(model_base_path, "models", "weights_19", "depth.pth")
    
    # 모델 파일 경로
    encoder_module_path = os.path.join(model_base_path, f"{modelname}_encoder.py")
    decoder_module_path = os.path.join(model_base_path, f"{modelname}_decoder.py")
    
    # 파일 존재 확인
    assert os.path.exists(encoder_path), f"Encoder weights not found: {encoder_path}"
    assert os.path.exists(decoder_path), f"Decoder weights not found: {decoder_path}"
    assert os.path.exists(encoder_module_path), f"Encoder module not found: {encoder_module_path}"
    assert os.path.exists(decoder_module_path), f"Decoder module not found: {decoder_module_path}"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from", device)
    print("-> Encoder path:", encoder_path)
    print("-> Decoder path:", decoder_path)
    print("-> Encoder module path:", encoder_module_path)
    print("-> Decoder module path:", decoder_module_path)

    # 동적으로 모듈 로드
    encoder_module = load_module_from_path(f"{modelname}_encoder", encoder_module_path)
    decoder_module = load_module_from_path(f"{modelname}_decoder", decoder_module_path)

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    # LOADING PRETRAINED MODEL - 절대 경로에서 동적으로 로드
    print("   Loading pretrained encoder from:", encoder_module_path)
    encoder = encoder_module.LiteMono(model=args.model,
                                      height=feed_height,
                                      width=feed_width)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder from:", decoder_module_path)
    depth_decoder = decoder_module.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()
    
    '''
    # 하위 디렉토리 많을때
    '''
    for image_folder in args.image_path:
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

    

def direct():

    directory = r"C:\Users\wodud\OneDrive\Desktop\도로주행 데이터\v4.1" 

    folders = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    print("디렉토리 내 폴더 목록:")
    for folder in folders:
        print(folder)
    
    return folders


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)