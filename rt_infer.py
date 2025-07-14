import datetime
import os
import re
import shutil
import time
from glob import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torch.profiler
from thop import profile, clever_format

# --- (기존 상수 및 클래스 정의는 동일하게 유지) ---
MODEL_WEIGHTS_FOLDER = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\lite-mono_640x192"
# MODEL_WEIGHTS_FOLDER = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\halfdims\models\weights_9"
# MODEL_WEIGHTS_FOLDER = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\encoder"
# MODEL_WEIGHTS_FOLDER = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\CBAM\models\weights_29"
MODEL_ENCODER_FILE = "encoder.pth"
MODEL_DECODER_FILE = "depth.pth"
MODEL_LITEMONO_CLASS_NAME = "lite-mono"
MODEL_TAG = "lite"

IMAGE_FOLDER = r"C:\Users\wodud\OneDrive\Desktop\도로주행 데이터\2011_09_26_drive_0009_sync_학습함"
NETWORKS_MODULE_PATH = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom"
PROFILER_LOG_DIR = "./log_dir_pytorch_single_model_final"

FEED_HEIGHT = 192
FEED_WIDTH = 640
BATCH_SIZE = 1 # FLOPs 계산 및 프로파일링용. 실제 벤치마크는 이미지별로 진행
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 20 # 개별 이미지당 측정 반복 횟수
PROFILER_ACTIVE_ITERATIONS = 5

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
    import networks

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
        # print(f"   {runner_name} (개별 이미지) 웜업 ({warmup_iterations}회)...") # 개별 이미지 웜업 로그는 너무 많을 수 있어 주석 처리
        for _ in range(warmup_iterations):
            with torch.no_grad(): _ = pytorch_model(input_data_torch)
            if device_torch.type == 'cuda': torch.cuda.synchronize()

    # print(f"   {runner_name} (개별 이미지) 추론 ({num_iterations}회)...") # 개별 이미지 추론 로그는 너무 많을 수 있어 주석 처리
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
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_trace_dir = os.path.join(log_dir, "plugins", "profile", now_str)
    os.makedirs(tb_trace_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n--- {model_name_str} PyTorch 레이어별 프로파일링 시작 ({active_iterations}회 반복) ---")
    
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device_torch.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=active_iterations, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_trace_dir),
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

    # TensorBoard trace 생성
    try:
        trace_dir = os.path.join(log_dir, "plugins", "profile")
        latest_run_dir = max(glob(os.path.join(trace_dir, "*")), key=os.path.getmtime)
        trace_json_files = glob(os.path.join(latest_run_dir, "*.pt.trace.json"))
        for f in trace_json_files:
            if re.search(r"\.pt\.trace\.json$", f):
                src = f
                dst = os.path.join(latest_run_dir, "local.trace")
                shutil.copyfile(src, dst)
                print(f"✔️ TensorBoard용 local.trace 생성 완료 → {dst}")
                break
        else:
            print("⚠️ .pt.trace.json 파일이 발견되지 않았습니다.")
    except Exception as e:
        print(f"❌ local.trace 복사 중 예외 발생: {e}")

    # 🎯 핵심: PyTorch Profiler 테이블을 CSV로 변환
    csv_save_path = os.path.join(log_dir, f"{model_name_str}_profile_summary.csv")
    
    try:
        import csv
        
        # PyTorch Profiler 테이블 문자열 생성 (전체 테이블, 제한 없음)
        sort_key = "self_cuda_time_total" if device_torch.type == 'cuda' else "self_cpu_time_total"
        
        # 전체 테이블을 문자열로 가져오기 (row_limit=None으로 모든 행 포함)
        try:
            table_str = prof.key_averages().table(sort_by=sort_key, row_limit=None)
        except:
            try:
                table_str = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=None)
            except:
                table_str = prof.key_averages().table(row_limit=None)
        
        # 테이블 문자열을 파싱하여 CSV 데이터로 변환
        lines = table_str.strip().split('\n')
        
        # 헤더와 구분선 찾기
        header_line = None
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if 'Name' in line and 'Self CPU' in line:
                header_line = line
                # 다음 구분선을 찾아서 데이터 시작점 결정
                for j in range(i+1, len(lines)):
                    if '----' in lines[j]:
                        data_start_idx = j + 1
                        break
                break
        
        if header_line is None:
            print("❌ 테이블 헤더를 찾을 수 없습니다.")
            return
            
        # 헤더 파싱 (공백으로 구분된 컬럼들)
        # 정규식으로 컬럼 경계를 찾기
        import re
        header_parts = re.split(r'\s{2,}', header_line.strip())
        
        # 데이터 행들 파싱
        csv_rows = []
        csv_rows.append(header_parts)  # 헤더 추가
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if not line or '----' in line:
                continue
                
            # 각 행을 동일한 방식으로 파싱
            row_parts = re.split(r'\s{2,}', line)
            
            # 헤더와 컬럼 수가 일치하도록 조정
            while len(row_parts) < len(header_parts):
                row_parts.append('')
            if len(row_parts) > len(header_parts):
                row_parts = row_parts[:len(header_parts)]
                
            csv_rows.append(row_parts)
        
        # CSV 파일로 저장
        with open(csv_save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        
        if os.path.exists(csv_save_path):
            file_size = os.path.getsize(csv_save_path)
            print(f"✅ 전체 프로파일링 테이블 CSV 저장 완료!")
            print(f"   파일 경로: {csv_save_path}")
            print(f"   파일 크기: {file_size} bytes")
            print(f"   총 행 수: {len(csv_rows)} (헤더 포함)")
        else:
            print("❌ CSV 파일이 생성되지 않았습니다!")
            
    except Exception as e:
        print(f"❌ CSV 저장 중 오류: {e}")
        import traceback
        traceback.print_exc()

    # 🎯 전체 프로파일링 테이블 출력 (모든 77개 레이어)
    print(f"\n{model_name_str} 전체 프로파일링 결과 ({sort_key} 기준 정렬):")
    print("=" * 150)
    try:
        # 전체 테이블 출력 (row_limit=None으로 모든 행 표시)
        print(prof.key_averages().table(sort_by=sort_key, row_limit=None))
    except Exception as e:
        print(f"프로파일러 테이블 출력 오류 (sort_by='{sort_key}'): {e}")
        try:
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=None))
        except Exception as e_cpu:
            print(f"CPU 기준 출력도 실패: {e_cpu}")
            print(prof.key_averages().table(row_limit=None))
    print("=" * 150)

    print(f"TensorBoard 실행 예: tensorboard --logdir=\"{log_dir}\"")
    print(f"--------------------------------------------------")

def calculate_model_flops_params(model_instance: LiteMonoComplete, input_tensor_shape: tuple, device_torch: torch.device):
    # (calculate_model_flops_params 함수 내용은 이전과 동일)
    dummy_input = torch.randn(input_tensor_shape).to(device_torch)
    total_flops, total_params = profile(model_instance, inputs=(dummy_input,), verbose=False)
    return total_flops, total_params

def benchmark_folder_once(
    model_tag: str,
    pytorch_model: LiteMonoComplete,
    image_paths: list,
    transform: T.Compose,
    device_torch: torch.device,
    benchmark_iters: int,
    warmup_iters: int
) -> float:
    print(f"\n>>> (폴더 벤치마크) '{model_tag}' 모델 - 이미지 총 {len(image_paths)}개 처리 시작")

    BATCH_SIZE = 1
    batched_avg_times = []
    num_images = len(image_paths)
    num_processed_images = 0

    for i in range(0, num_images, BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = transform(img)
                batch_images.append(tensor)
            except Exception as e:
                print(f"[오류 - {model_tag}] - 이미지 {os.path.basename(p)} 로드 실패: {e}")

        if not batch_images:
            continue

        x_batch = torch.stack(batch_images).to(device_torch)

        avg_batch_time = run_pytorch_benchmark(
            model_tag, pytorch_model, x_batch,
            benchmark_iters, warmup_iters, device_torch
        )

        avg_per_image = avg_batch_time / len(batch_images)
        batched_avg_times.extend([avg_per_image] * len(batch_images))
        num_processed_images += len(batch_images)

    if batched_avg_times:
        mean_folder_inference_time = np.mean(batched_avg_times)
        print(f"  처리된 이미지 수: {num_processed_images}")
        print(f"  폴더 전체 평균 추론 시간: {mean_folder_inference_time:.2f} ms/장")
        return mean_folder_inference_time
    else:
        print(f"[정보 - {model_tag}] 벤치마크를 위한 이미지를 처리하지 못했습니다.")
        return -1.0

def main():
    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device_torch = torch.device("cpu")
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

    input_shape_for_flops = (1, 3, FEED_HEIGHT, FEED_WIDTH) # BATCH_SIZE를 1로 고정 (FLOPs 계산용)

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

    # 2. FLOPs 및 파라미터 계산 (최초 1회만 수행)
    model_flops = -1
    model_params = -1
    try:
        model_flops, model_params = calculate_model_flops_params(pytorch_model, input_shape_for_flops, device_torch)
        flops_str, params_str = clever_format([model_flops, model_params], "%.3f")
        print(f"[정보 - {model_tag_for_analysis}] FLOPs: {flops_str} | Params: {params_str}")
    except Exception as e:
        print(f"[오류 - {model_tag_for_analysis}] FLOPs/Params 계산 실패: {e}")

    # 3. PyTorch 레이어별 프로파일링 (최초 1회, 첫 번째 이미지 사용)
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

    # --- 👇 (C) 폴더 전체 벤치마크 4회 반복 ---
    print(f"\n>>> (C) {model_tag_for_analysis} 전체 폴더 벤치마크 4회 반복 시작")
    num_folder_benchmarks = 4
    all_folder_avg_times = []

    for i in range(num_folder_benchmarks):
        print(f"\n--- 폴더 벤치마크 반복 {i+1}/{num_folder_benchmarks} ---")
        folder_avg_time = benchmark_folder_once(
            model_tag_for_analysis,
            pytorch_model,
            img_paths,
            transform,
            device_torch,
            BENCHMARK_ITERATIONS, # 개별 이미지당 반복 횟수
            WARMUP_ITERATIONS     # 개별 이미지당 웜업 횟수
        )
        if folder_avg_time >= 0: # 유효한 결과인 경우에만 추가
            all_folder_avg_times.append(folder_avg_time)

    # 최종 결과 요약
    print(f"\n\n===== {model_tag_for_analysis} 모델 반복 벤치마크 최종 요약 =====")
    if model_flops != -1 and model_params != -1:
        flops_str_final, params_str_final = clever_format([model_flops, model_params], "%.3f")
        print(f"  FLOPs           : {flops_str_final}")
        print(f"  Params          : {params_str_final}")
    else:
        print(f"  FLOPs           : N/A (계산 실패)")
        print(f"  Params          : N/A (계산 실패)")

    if all_folder_avg_times:
        print("\n  각 폴더 벤치마크 반복별 평균 추론 시간:")
        for i, t_avg in enumerate(all_folder_avg_times):
            print(f"    반복 {i+1}: {t_avg:.2f} ms/장")
        
        overall_mean_of_folder_benchmarks = np.mean(all_folder_avg_times)
        print(f"\n  총 {len(all_folder_avg_times)}회 폴더 벤치마크의 전체 평균 추론 시간: {overall_mean_of_folder_benchmarks:.2f} ms/장")

        if model_flops > 0 and overall_mean_of_folder_benchmarks > 0:
            gflops_per_sec = (model_flops / (overall_mean_of_folder_benchmarks / 1000)) / 1e9
            print(f"  성능 (GFLOPs/sec, 전체 평균 기준): {gflops_per_sec:.2f}")
    else:
        print(f"[정보 - {model_tag_for_analysis}] 폴더 벤치마크를 성공적으로 완료하지 못했습니다.")

    print(f"===== {model_tag_for_analysis} 모델 분석 완료 =====")
    print("\n\n========= ✅ 전체 분석 완료 =========")
    if PROFILER_LOG_DIR and os.path.exists(PROFILER_LOG_DIR):
        print("PyTorch Profiler 결과는 TensorBoard로 확인하세요:")
        print(f"  tensorboard --logdir=\"{PROFILER_LOG_DIR}\"")

if __name__ == "__main__":
    main()