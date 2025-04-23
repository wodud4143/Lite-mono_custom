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
                        default= r"C:\Users\wodud\OneDrive\Desktop\sample\test2_NO_dil\frame_00375.jpg" #direct()
                        )# required=True

    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use',
                        default=r'C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\no_dilation\models\weights_40'
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





class Showrecunstruction:
    def __init__(self, args ,options):
        self.opt = options
        self.args = args
        
     
    # region  predict_poses  
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models_pose["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs   
        
        
    # region generate_images_pred      
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    
        
        
    # region test_simple
    def test_simple(self):
        """Function to predict for a single image or folder of images
        """
        assert self.args.load_weights_folder is not None, \
            "You must specify the --load_weights_folder parameter"

        if torch.cuda.is_available() and not self.args.no_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print("-> Loading model from ", self.args.load_weights_folder)
        encoder_path = os.path.join(self.args.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(self.args.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)

        # extract the height and width of image that this model was trained with
        feed_height = encoder_dict['height']
        feed_width = encoder_dict['width']

        # LOADING PRETRAINED MODEL
        print("   Loading pretrained encoder")
        encoder = networks.LiteMono(model=self.args.model,
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
        
        
        # for image_folder in args.image_path:
        image_folder = r"C:\Users\wodud\OneDrive\Desktop\sample\test2_NO_dil"
        if image_folder :
            # FINDING INPUT IMAGES
            if os.path.isfile(image_folder) and not self.args.test:
                # Only testing on a single image
                paths = [image_folder]
                output_directory = os.path.dirname(image_folder)
            elif os.path.isfile(image_folder) and self.args.test:
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
                paths = glob.glob(os.path.join(image_folder, '*.{}'.format(self.args.ext)))
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

                    # region PREDICTION
                    input_image = input_image.to(device)
                    features = encoder(input_image)
                    outputs = depth_decoder(features)
                    
                    outputs.update(self.predict_poses(inputs, features))
                    
                    self.generate_images_pred(input_image,outputs)
                    

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
   


if __name__ == '__main__':
    options = LiteMonoOptions()
    opts = options.parse()
    args = parse_args()
    showrecunstruction = Showrecunstruction(args,opts)
    showrecunstruction.test_simple()
    
