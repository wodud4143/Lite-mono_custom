# from __future__ import absolute_import, division, print_function

# import os
# import sys
# import numpy as np
# import PIL.Image as pil
# import matplotlib as mpl
# import matplotlib.cm as cm
# import cv2
# import time
# import argparse

# import torch
# from torchvision import transforms

# import networks
# from layers import disp_to_depth


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Real-time webcam depth estimation with Lite-Mono models.')

#     parser.add_argument('--load_weights_folder', type=str,
#                         help='path of a pretrained model to use',
#                         default=r'C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\Ghost_Cooratt_100ep_pretrain\models\weights_93'
#                         )

#     parser.add_argument('--model', type=str,
#                         default="lite-mono",
#                         choices=[
#                             "lite-mono",
#                             "lite-mono-small",
#                             "lite-mono-tiny",
#                             "lite-mono-8m"])

#     parser.add_argument('--camera_id', type=int,
#                         help='camera device id (default: 0)', default=0)
                        
#     parser.add_argument('--width', type=int,
#                         help='input width for the network (default: 640)', default=640)
                        
#     parser.add_argument('--height', type=int,
#                         help='input height for the network (default: 192)', default=192)
                        
#     parser.add_argument('--fps_avg_frame_count', type=int,
#                         help='frames for FPS averaging (default: 10)', default=10)
                        
#     parser.add_argument("--no_cuda",
#                         help='if set, disables CUDA',
#                         action='store_true')
                        
#     parser.add_argument("--show_input",
#                         help='if set, shows input image alongside depth map',
#                         action='store_true')
                        
#     parser.add_argument("--save_output",
#                         help='if set, saves output frames to a video file',
#                         action='store_true')
                        
#     parser.add_argument("--output_dir", type=str,
#                         help='directory to save output frames',
#                         default="output_webcam")

#     return parser.parse_args()


# def webcam_depth_estimation():
#     """Real-time depth estimation from webcam feed using Lite-Mono
#     """
#     args = parse_args()
    
#     assert args.load_weights_folder is not None, \
#         "You must specify the --load_weights_folder parameter"

#     # Check if CUDA is available
#     if torch.cuda.is_available() and not args.no_cuda:
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     print(f"-> Using device: {device}")
    
#     # Create output directory if saving outputs
#     if args.save_output:
#         os.makedirs(args.output_dir, exist_ok=True)
#         video_path = os.path.join(args.output_dir, f"depth_estimation_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
#         print(f"-> Saving output to: {video_path}")

#     # Load model weights
#     print("-> Loading model from", args.load_weights_folder)
#     encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
#     decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

#     encoder_dict = torch.load(encoder_path, map_location=device)
#     decoder_dict = torch.load(decoder_path, map_location=device)

#     # Extract model dimensions
#     feed_height = args.height
#     feed_width = args.width
#     if 'height' in encoder_dict and 'width' in encoder_dict:
#         print(f"-> Using model feed dimensions: {encoder_dict['width']}x{encoder_dict['height']}")
#         feed_height = encoder_dict['height']
#         feed_width = encoder_dict['width']
#     else:
#         print(f"-> Using specified feed dimensions: {feed_width}x{feed_height}")

#     # Load encoder
#     print("-> Loading pretrained encoder")
#     encoder = networks.LiteMono(model=args.model,
#                                 height=feed_height,
#                                 width=feed_width)

#     model_dict = encoder.state_dict()
#     encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
#     encoder.to(device)
#     encoder.eval()

#     # Load decoder
#     print("-> Loading pretrained decoder")
#     depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
#     depth_model_dict = depth_decoder.state_dict()
#     depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
#     depth_decoder.to(device)
#     depth_decoder.eval()

#     # Initialize webcam
#     print(f"-> Initializing webcam (ID: {args.camera_id})")
#     cap = cv2.VideoCapture(args.camera_id)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     # Set camera resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     # Get actual camera resolution (may differ from requested)
#     cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"-> Camera resolution: {cam_width}x{cam_height}")
    
#     # Initialize video writer if saving output
#     video_writer = None
#     if args.save_output:
#         if args.show_input:
#             # Combined view will be side by side
#             output_width = cam_width * 2
#             output_height = cam_height
#         else:
#             # Only depth map
#             output_width = cam_width
#             output_height = cam_height
            
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video_writer = cv2.VideoWriter(video_path, fourcc, 30, 
#                                       (output_width, output_height))

#     # FPS calculation variables
#     counter, fps = 0, 0
#     start_time = time.time()
#     fps_avg_frame_count = args.fps_avg_frame_count
    
#     print("-> Press 'q' to quit, 's' to save a single frame")
    
#     try:
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 print("Error: Failed to capture image from camera.")
#                 break
                
#             counter += 1
            
#             # Convert from BGR to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Convert to PIL Image for processing
#             input_image = pil.fromarray(frame_rgb)
#             original_width, original_height = input_image.size
            
#             # Prepare image for model (resize, convert to tensor)
#             input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
#             input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0)
            
#             # Process with model
#             with torch.no_grad():
#                 input_tensor = input_tensor.to(device)
#                 features = encoder(input_tensor)
#                 outputs = depth_decoder(features)
                
#                 disp = outputs[("disp", 0)]
                
#                 # Resize to original resolution
#                 disp_resized = torch.nn.functional.interpolate(
#                     disp, (original_height, original_width), mode="bilinear", align_corners=False)
                
#                 # Convert to depth
#                 scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                
#                 # Convert to numpy for visualization
#                 disp_resized_np = disp_resized.squeeze().cpu().numpy()
                
#                 # Create color-mapped visualization
#                 vmax = np.percentile(disp_resized_np, 95)
#                 normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
#                 mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
#                 colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                
#                 # Convert back to BGR for OpenCV display
#                 depth_colormap_bgr = cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR)
                
#                 # Calculate FPS
#                 if counter % fps_avg_frame_count == 0:
#                     end_time = time.time()
#                     fps = fps_avg_frame_count / (end_time - start_time)
#                     start_time = time.time()
                
#                 # Add FPS info to depth image
#                 cv2.putText(depth_colormap_bgr, f"FPS: {fps:.1f}", (10, 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
#                 # Prepare display output
#                 if args.show_input:
#                     # Show input image and depth side by side
#                     combined_output = np.hstack((frame, depth_colormap_bgr))
#                     cv2.imshow('Webcam Depth Estimation (Input | Depth)', combined_output)
                    
#                     # Save output if requested
#                     if video_writer is not None:
#                         video_writer.write(combined_output)
#                 else:
#                     # Show only depth
#                     cv2.imshow('Webcam Depth Estimation', depth_colormap_bgr)
                    
#                     # Save output if requested
#                     if video_writer is not None:
#                         video_writer.write(depth_colormap_bgr)
            
#             # Check for key presses
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 print("-> Quitting")
#                 break
#             elif key == ord('s'):
#                 # Save current frame
#                 timestamp = time.strftime("%Y%m%d_%H%M%S")
#                 save_dir = args.output_dir
#                 os.makedirs(save_dir, exist_ok=True)
                
#                 # Save depth image
#                 depth_path = os.path.join(save_dir, f"depth_{timestamp}.jpg")
#                 cv2.imwrite(depth_path, depth_colormap_bgr)
                
#                 # Save original image
#                 rgb_path = os.path.join(save_dir, f"frame_{timestamp}.jpg")
#                 cv2.imwrite(rgb_path, frame)
                
#                 # Save depth data as numpy file
#                 npy_path = os.path.join(save_dir, f"depth_{timestamp}.npy")
#                 np.save(npy_path, scaled_disp.cpu().numpy())
                
#                 print(f"-> Saved frame to {save_dir}")
    
#     except KeyboardInterrupt:
#         print("-> Interrupted by user")
    
#     finally:
#         # Clean up
#         if video_writer is not None:
#             video_writer.release()
#         cap.release()
#         cv2.destroyAllWindows()
#         print("-> Done")


# if __name__ == '__main__':
#     webcam_depth_estimation()


import numpy as np

data = np.load('0000000038_disp.npy')

a = data