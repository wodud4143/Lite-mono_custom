import os
import time
from glob import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torch.profiler
from thop import profile, clever_format


MODEL_WEIGHTS_FOLDER = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\lite-mono_640x192"
MODEL_ENCODER_FILE = "encoder.pth"
MODEL_DECODER_FILE = "depth.pth"
MODEL_LITEMONO_CLASS_NAME = "lite-mono"
MODEL_TAG = "original" # 프로파일링 및 결과 출력 시 사용할 태그


IMAGE_FOLDER   = r"C:\Users\wodud\OneDrive\Desktop\도로주행 데이터\2011_09_26_drive_0009_sync_학습함"
NETWORKS_MODULE_PATH = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom"
PROFILER_LOG_DIR = "./log_dir_pytorch_single_model_final" 

FEED_HEIGHT = 192
FEED_WIDTH  = 640
BATCH_SIZE  = 1
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 20 # 각 이미지당 측정 반복 횟수
PROFILER_ACTIVE_ITERATIONS = 5 # PyTorch Profiler용 측정 반복 횟수


class LiteMonoComplete(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs[("disp", 0)]

def load_specific_pytorch_model(
    weights_folder_for_encoder: str,
    encoder_filename: str,
    weights_folder_for_decoder: str,
    decoder_filename: str,
    litemono_class_model_name: str,
    feed_height: int,
    feed_width: int,
    device: torch.device
) -> LiteMonoComplete:
    import sys
    if NETWORKS_MODULE_PATH not in sys.path:
        sys.path.insert(0, NETWORKS_MODULE_PATH)
    import networks # 사용자 정의 networks 모듈

    encoder_path = os.path.join(weights_folder_for_encoder, encoder_filename)
    decoder_path = os.path.join(weights_folder_for_decoder, decoder_filename)

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"no encoder {encoder_path}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"no decoder {decoder_path}")

    print(f"  인코더 ({litemono_class_model_name}): {encoder_path}")
    encoder_dict = torch.load(encoder_path, map_location=device)

    print(f"  디코더: {decoder_path}")
    decoder_dict = torch.load(decoder_path, map_location=device)

    encoder = networks.LiteMono(model=litemono_class_model_name, height=feed_height, width=feed_width)
    enc_state = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in enc_state and enc_state[k].shape == v.shape}, strict=False)

    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    dec_state = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in dec_state and dec_state[k].shape == v.shape}, strict=False)

    model = LiteMonoComplete(encoder, depth_decoder)
    model.to(device)
    model.eval()
    return model


def run_pytorch_benchmark(
    runner_name: str,
    pytorch_model: LiteMonoComplete,
    input_data_torch: torch.Tensor,
    num_iterations: int,
    warmup_iterations: int,
    device_torch: torch.device
    ):
    total_time_ms = 0.0

    if warmup_iterations > 0:
        print(f"  {runner_name} 웜업 ({warmup_iterations}회)...")
        for _ in range(warmup_iterations):
            with torch.no_grad(): _ = pytorch_model(input_data_torch)
            if device_torch.type == 'cuda': torch.cuda.synchronize()

    print(f"  {runner_name} 추론 ({num_iterations}회)...")
    for i in range(num_iterations):
        start_time = time.perf_counter()
        with torch.no_grad():

            _ = pytorch_model(input_data_torch)
        if device_torch.type == 'cuda': torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time_ms += (end_time - start_time) * 1000

    avg_time_ms = total_time_ms / num_iterations if num_iterations > 0 else 0

    return avg_time_ms


def profile_pytorch_model_layers(
    model_name_str: str,
    model: LiteMonoComplete,
    input_data_torch: torch.Tensor,
    device_torch: torch.device,
    active_iterations: int,
    log_dir: str
    ):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    profiler_output_dir = os.path.join(log_dir, model_name_str)
    if not os.path.exists(profiler_output_dir):
        os.makedirs(profiler_output_dir, exist_ok=True)

    print(f"\n--- {model_name_str} PyTorch 레이어별 프로파일링 시작 ({active_iterations}회 반복) ---")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device_torch.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=active_iterations, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(1 + 1 + active_iterations):
            with torch.no_grad():
                _ = model(input_data_torch)
            if device_torch.type == 'cuda':
                torch.cuda.synchronize()
            prof.step()

    sort_key = "self_cuda_time_total" if device_torch.type == 'cuda' else "self_cpu_time_total"
    print(f"{model_name_str} 프로파일링 요약 (상위 10개 {sort_key} 기준):")
    try:
        print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
    except Exception as e:
        print(f"프로파일러 테이블 출력 오류 (sort_by='{sort_key}'): {e}. CPU 시간 기준으로 재시도합니다.")
        try:
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        except Exception as e_cpu:
            print(f"CPU 시간 기준 테이블 출력도 실패: {e_cpu}. 기본 테이블을 출력합니다.")
            print(prof.key_averages().table(row_limit=10))


    print(f"TensorBoard 로그 저장 위치: {profiler_output_dir}")
    print(f"TensorBoard 실행 예: tensorboard --logdir=\"{log_dir}\"")
    print(f"--------------------------------------------------")


def calculate_model_flops_params(model_instance: LiteMonoComplete, input_tensor_shape: tuple, device_torch: torch.device):
    dummy_input = torch.randn(input_tensor_shape).to(device_torch)
    total_flops, total_params = profile(model_instance, inputs=(dummy_input,), verbose=False)
    return total_flops, total_params

def main():
    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[정보] PyTorch Device: {device_torch}")

    img_paths = sorted(glob(os.path.join(IMAGE_FOLDER, "*.png")))
    if not img_paths:
        print(f"[오류] 이미지 폴더에 .png 파일이 없습니다: {IMAGE_FOLDER}")
        return
    print(f"[정보] 총 {len(img_paths)}개의 PNG 이미지를 사용하여 벤치마크 및 프로파일링을 수행합니다.\n")

    transform = T.Compose([
        T.Resize((FEED_HEIGHT, FEED_WIDTH)),
        T.ToTensor(),
    ])

    input_shape_for_flops = (BATCH_SIZE, 3, FEED_HEIGHT, FEED_WIDTH)


    model_tag_for_analysis = MODEL_TAG
    print(f"\n\n===== {model_tag_for_analysis} 모델 분석 시작 =====")

    print(f"\n>>> (A) {model_tag_for_analysis} 모델 로드")
    try:
        pytorch_model = load_specific_pytorch_model(
            MODEL_WEIGHTS_FOLDER,
            MODEL_ENCODER_FILE,
            MODEL_WEIGHTS_FOLDER, 
            MODEL_DECODER_FILE,
            MODEL_LITEMONO_CLASS_NAME,
            FEED_HEIGHT, FEED_WIDTH, device_torch
        )
    except Exception as e:
        print(f"[오류] {model_tag_for_analysis} 모델 로드 실패: {e}")
        return

    # 2. FLOPs 및 파라미터 계산
    model_flops = -1
    model_params = -1
    try:
        model_flops, model_params = calculate_model_flops_params(pytorch_model, input_shape_for_flops, device_torch)
        flops_str, params_str = clever_format([model_flops, model_params], "%.3f")
        print(f"[정보 - {model_tag_for_analysis}] FLOPs: {flops_str} | Params: {params_str}")
    except Exception as e:
        print(f"[오류 - {model_tag_for_analysis}] FLOPs/Params 계산 실패: {e}")


    # 3. PyTorch 레이어별 프로파일링
    if img_paths:
        print(f"\n>>> (B) {model_tag_for_analysis} 레이어별 프로파일링 (첫 번째 이미지 사용)")
        img_for_profile = Image.open(img_paths[0]).convert("RGB")
        x_torch_for_profile = transform(img_for_profile).unsqueeze(0).to(device_torch)
        try:
            profile_pytorch_model_layers(
                model_tag_for_analysis, pytorch_model, x_torch_for_profile, device_torch,
                PROFILER_ACTIVE_ITERATIONS, PROFILER_LOG_DIR
            )
        except Exception as e:
            print(f"[오류 - {model_tag_for_analysis}] 레이어 프로파일링 실패: {e}")
    else:
        print("[정보] 프로파일링을 위한 이미지가 없습니다.")


    # 4. 모든 이미지에 대한 벤치마크
    print(f"\n>>> (C) {model_tag_for_analysis} 전체 이미지 벤치마크")
    model_all_avg_times = []
    num_processed_images = 0

    for i, img_path in enumerate(img_paths):
        print(f"  이미지 {i+1}/{len(img_paths)} 벤치마크 중: {os.path.basename(img_path)}...")
        try:
            img = Image.open(img_path).convert("RGB")
            x_torch_input = transform(img).unsqueeze(0).to(device_torch)

            avg_time_ms = run_pytorch_benchmark(
                model_tag_for_analysis, pytorch_model, x_torch_input,
                BENCHMARK_ITERATIONS, WARMUP_ITERATIONS, device_torch
            )
            model_all_avg_times.append(avg_time_ms)
            num_processed_images +=1
        except Exception as e:
            print(f"[오류] {model_tag_for_analysis} - 이미지 {os.path.basename(img_path)} 벤치마크 중 문제 발생: {e}")
            continue

    if num_processed_images > 0:
        mean_inference_time = np.mean(model_all_avg_times)
        gflops_per_sec = 0
        if model_flops > 0 and mean_inference_time > 0:
            gflops_per_sec = (model_flops / (mean_inference_time / 1000)) / 1e9

        print(f"\n[최종 결과 - {model_tag_for_analysis}]")
        print(f"  처리된 이미지 수: {num_processed_images}")

        if model_flops != -1 and model_params != -1:
            flops_str_final, params_str_final = clever_format([model_flops, model_params], "%.3f")
            print(f"  FLOPs           : {flops_str_final}")
            print(f"  Params          : {params_str_final}")
        else:
            print(f"  FLOPs           : N/A (계산 실패)")
            print(f"  Params          : N/A (계산 실패)")

        print(f"  평균 추론 시간  : {mean_inference_time:.2f} ms/장")
        if model_flops > 0 :
             print(f"  성능 (GFLOPs/sec): {gflops_per_sec:.2f}")
    else:
        print(f"[정보 - {model_tag_for_analysis}] 벤치마크를 위한 이미지를 처리하지 못했습니다.")
    print(f"===== {model_tag_for_analysis} 모델 분석 완료 =====")


    print("\n\n========= ✅ 전체 분석 완료 =========")
    print("PyTorch Profiler 결과는 TensorBoard로 확인하세요:")
    print(f"  tensorboard --logdir=\"{PROFILER_LOG_DIR}\"")


if __name__ == "__main__":
    main()