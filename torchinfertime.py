# measure_pth_latency.py
import os
import os.path as osp
import time
import glob
import cv2
import numpy as np
import torch

from torch.utils.data import DataLoader
import networks
import datasets


DATA_DIR         = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\kitti_data\2011_09_26\2011_09_26_drive_0001_sync\image_02\data"
ENCODER_PATH     = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\v4_3_R_a3_gm_cutout\models\weights_49\encoder.pth"
DECODER_PATH     = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\v4_3_R_a3_gm_cutout\models\weights_49\depth.pth"
MODEL_TYPE       = "litemono"               
IMG_SIZE         = (640, 192)            
NUM_OUTER_LOOPS  = 20
WARMUP_LOOPS     = 5                    
USE_FP16         = False                
DEVICE           = "cuda"

torch.backends.cudnn.benchmark = True     

def load_models(encoder_path, decoder_path):
  
    encoder_ckpt = torch.load(encoder_path, map_location="cpu")
    decoder_ckpt = torch.load(decoder_path, map_location="cpu")

    h = int(encoder_ckpt.get("height", IMG_SIZE[1]))
    w = int(encoder_ckpt.get("width",  IMG_SIZE[0]))

    encoder = networks.LiteMono(model="lite-mono",
                                height=h, width=w)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))

    enc_state = encoder.state_dict()
    dec_state = depth_decoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_ckpt.items() if k in enc_state}, strict=False)
    depth_decoder.load_state_dict({k: v for k, v in decoder_ckpt.items() if k in dec_state}, strict=False)

    encoder.to(DEVICE).eval()
    depth_decoder.to(DEVICE).eval()
    return encoder, depth_decoder, (h, w)

def preprocess_image_cv2(path, size_wh):
    """ TRT 코드와 동일 개념: BGR->RGB, resize(W,H), [0,1], CHW, float32, 1D 아님(텐서로 유지) """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size_wh)                   # (W,H)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))               # CHW
    tensor = torch.from_numpy(img).unsqueeze(0)      # NCHW
    return tensor

@torch.inference_mode()
def measure_latency_single_images(encoder, decoder, image_paths, size_wh, use_fp16=False):
    """ TRT 스크립트와 동일하게 파일 리스트를 순회하며 per-image forward 시간을 측정 """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    times_ms = []

    for outer in range(1, NUM_OUTER_LOOPS + 1):
        if outer == 1:
            print("웜업중")

        for p in image_paths:
     
            inp = preprocess_image_cv2(p, size_wh).pin_memory()
            inp = inp.to(DEVICE, non_blocking=True)

  
            if use_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    torch.cuda.synchronize()
                    start_event.record()
                    feats = encoder(inp)
                    _ = decoder(feats)
                    end_event.record()
                    torch.cuda.synchronize()
            else:
                torch.cuda.synchronize()
                start_event.record()
                feats = encoder(inp)
                _ = decoder(feats)
                end_event.record()
                torch.cuda.synchronize()

            # 이벤트 간 경과(ms)
            elapsed_ms = start_event.elapsed_time(end_event)

            if outer > WARMUP_LOOPS:
                times_ms.append(elapsed_ms)
                print(f"추론 시간: {elapsed_ms:.3f} ms")

    avg_ms = np.mean(times_ms) if times_ms else float("nan")
    print(f"{MODEL_TYPE} 평균 추론 시간: {avg_ms:.3f} ms")
    return avg_ms

def main():

    image_paths = sorted(glob.glob(osp.join(DATA_DIR, "*.png")))
    assert len(image_paths) > 0, "이미지 경로에 PNG가 없습니다."

    encoder, decoder, (h, w) = load_models(ENCODER_PATH, DECODER_PATH)


    size_wh = (w, h) if (w, h) != (IMG_SIZE[1], IMG_SIZE[0]) else IMG_SIZE

    measure_latency_single_images(
        encoder=encoder,
        decoder=decoder,
        image_paths=image_paths,
        size_wh=size_wh,
        use_fp16=USE_FP16
    )
    
def check_infertime(model_path):

    enc_model_path = model_path + "\encoder.pth"
    dec_model_path = model_path + "\depth.pth"     
    image_paths = sorted(glob.glob(osp.join(DATA_DIR, "*.png")))
    assert len(image_paths) > 0, "이미지 경로에 PNG가 없습니다."

    encoder, decoder, (h, w) = load_models(enc_model_path, dec_model_path)


    size_wh = (w, h) if (w, h) != (IMG_SIZE[1], IMG_SIZE[0]) else IMG_SIZE

    measure_latency_single_images(
        encoder=encoder,
        decoder=decoder,
        image_paths=image_paths,
        size_wh=size_wh,
        use_fp16=USE_FP16
    )
    

if __name__ == "__main__":
    main()
