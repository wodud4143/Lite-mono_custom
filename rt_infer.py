"""
TensorRT 엔진을 사용한 Lite-Mono 깊이 추정 (완전 수정 버전)
모든 오류 해결 및 최적화 완료
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import time

import torch
from torchvision import transforms

# TensorRT 관련
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TensorRTDepthEstimatorFixed:
    """TensorRT 엔진을 사용한 깊이 추정 클래스 (완전 수정)"""
    
    def __init__(self, engine_path):
        """
        TensorRT 엔진 로드
        
        Args:
            engine_path: TensorRT 엔진 파일 경로
        """
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 엔진 로드
        self._load_engine()
        
        # 입출력 정보 확인
        self._setup_bindings()
        
        # GPU 메모리 할당 초기화
        self._allocate_memory()
        
        print(f"✅ TensorRT 엔진 로드 완료: {engine_path}")
        print(f"입력: {self.input_name}, 크기: {self.input_shape}")
        print(f"출력: {self.output_name}, 크기: {self.output_shape}")
        
    def _load_engine(self):
        """TensorRT 엔진 로드"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"엔진 파일을 찾을 수 없습니다: {self.engine_path}")
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"TensorRT 엔진 로드 실패: {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
    
    def _setup_bindings(self):
        """입출력 바인딩 설정 (완전 수정)"""
        self.input_binding = None
        self.output_binding = None
        self.input_name = None
        self.output_name = None
        
        # TensorRT 버전별 호환성
        if hasattr(self.engine, 'num_io_tensors'):
            # TensorRT 10.x+
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_binding = i
                    self.input_name = name
                else:
                    self.output_binding = i
                    self.output_name = name
        else:
            # TensorRT 8.x/9.x
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    self.input_binding = i
                    self.input_name = name
                else:
                    self.output_binding = i
                    self.output_name = name
        
        # 기본 입출력 크기 설정 (필수!)
        self.base_input_shape = (-1, 3, 192, 640)
        self.base_output_shape = (-1, 1, 192, 640)
        
        # 초기 shape 설정 (배치 크기 1로 시작)
        self.input_shape = (1, 3, 192, 640)
        self.output_shape = (1, 1, 192, 640)
        
        # 바인딩 검증
        if self.input_name is None or self.output_name is None:
            raise RuntimeError("입출력 바인딩을 찾을 수 없습니다")
        
    def _allocate_memory(self):
        """GPU 메모리 할당 초기화"""
        self.stream = cuda.Stream()
        self.d_input = None
        self.d_output = None
        self._last_batch_size = 0
        
    def _update_memory(self, batch_size):
        """배치 크기에 따른 메모리 업데이트 (PyCUDA 오류 해결)"""
        if batch_size != self._last_batch_size:
            # 기존 메모리 해제
            if self.d_input is not None:
                self.d_input.free()
            if self.d_output is not None:
                self.d_output.free()
            
            # 새로운 크기로 설정
            self.input_shape = (batch_size, 3, 192, 640)
            self.output_shape = (batch_size, 1, 192, 640)
            
            # PyCUDA 타입 오류 해결: 명시적으로 int 변환
            input_size = int(np.prod(self.input_shape)) * 4  # float32 = 4 bytes
            output_size = int(np.prod(self.output_shape)) * 4
            
            # GPU 메모리 할당
            try:
                self.d_input = cuda.mem_alloc(input_size)
                self.d_output = cuda.mem_alloc(output_size)
                print(f"메모리 할당 성공 - 배치: {batch_size}, 입력: {input_size//1024:.1f}KB, 출력: {output_size//1024:.1f}KB")
            except Exception as e:
                raise RuntimeError(f"GPU 메모리 할당 실패: {e}")
            
            self._last_batch_size = batch_size
            
            # 동적 shape 설정 (TensorRT 버전별 호환)
            try:
                if hasattr(self.engine, 'num_io_tensors'):
                    # TensorRT 10.x+
                    self.context.set_input_shape(self.input_name, self.input_shape)
                else:
                    # TensorRT 8.x/9.x
                    if hasattr(self.context, 'set_binding_shape'):
                        self.context.set_binding_shape(self.input_binding, self.input_shape)
            except Exception as e:
                print(f"⚠️ 동적 shape 설정 실패 (무시 가능): {e}")
    
    def predict(self, input_data):
        """
        TensorRT 추론 실행 (완전 수정)
        
        Args:
            input_data: numpy array (batch_size, 3, 192, 640)
            
        Returns:
            numpy array: disparity map (batch_size, 1, 192, 640)
        """
        # 입력 검증
        if not isinstance(input_data, np.ndarray):
            raise TypeError("입력 데이터는 numpy array여야 합니다")
        
        if len(input_data.shape) != 4:
            raise ValueError(f"입력 shape이 잘못되었습니다: {input_data.shape}, 기대값: (batch, 3, 192, 640)")
        
        batch_size = input_data.shape[0]
        
        # 데이터 타입 확인 및 변환
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # 메모리 업데이트 (배치 크기 변경시)
        self._update_memory(batch_size)
        
        try:
            # 입력 데이터를 GPU로 복사
            cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)
            
            # 추론 실행 (TensorRT 버전별 호환)
            success = False
            
            if hasattr(self.engine, 'num_io_tensors'):
                # TensorRT 10.x+
                self.context.set_tensor_address(self.input_name, int(self.d_input))
                self.context.set_tensor_address(self.output_name, int(self.d_output))
                success = self.context.execute_async_v3(self.stream.handle)
            else:
                # TensorRT 8.x/9.x
                bindings = [int(self.d_input), int(self.d_output)]
                
                if hasattr(self.engine, 'has_implicit_batch_dimension') and self.engine.has_implicit_batch_dimension:
                    success = self.context.execute_async(batch_size, bindings, self.stream.handle)
                else:
                    success = self.context.execute_async_v2(bindings, self.stream.handle)
            
            if not success:
                raise RuntimeError("TensorRT 추론 실행 실패")
            
            # 결과를 CPU로 복사
            output = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
            self.stream.synchronize()
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"TensorRT 추론 중 오류 발생: {e}")
    
    def __del__(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'd_input') and self.d_input:
                self.d_input.free()
            if hasattr(self, 'd_output') and self.d_output:
                self.d_output.free()
        except:
            pass  # 정리 중 오류 무시

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='TensorRT를 사용한 Lite-Mono 깊이 추정')

    parser.add_argument('--image_path', type=str,
                        help='이미지 파일 또는 폴더 경로',
                        default=direct())

    parser.add_argument('--engine_path', type=str,
                        help='TensorRT 엔진 파일 경로',
                        default='./tensorrt_output/litemono.engine')
    
    parser.add_argument('--output_dir', type=str,
                        help='결과 저장 디렉토리',
                        default='./tensorrt_depth_output')

    parser.add_argument('--ext', type=str,
                        help='이미지 확장자',
                        default="jpg")
    
    parser.add_argument('--batch_size', type=int,
                        help='배치 크기',
                        default=1)

    return parser.parse_args()

def preprocess_image(image_path, target_width=640, target_height=192):
    """
    이미지 전처리
    
    Args:
        image_path: 이미지 파일 경로
        target_width: 목표 너비
        target_height: 목표 높이
        
    Returns:
        tuple: (preprocessed_tensor, original_size)
    """
    try:
        # 이미지 로드
        image = pil.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # 리사이즈
        image_resized = image.resize((target_width, target_height), pil.LANCZOS)
        
        # 텐서 변환 및 정규화
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 1] 범위로 정규화
        ])
        
        image_tensor = transform(image_resized)  # (3, H, W)
        
        return image_tensor.numpy(), original_size
        
    except Exception as e:
        raise RuntimeError(f"이미지 처리 실패 {image_path}: {e}")

def postprocess_disparity(disparity, original_size):
    """
    Disparity 후처리
    
    Args:
        disparity: disparity map (H, W)
        original_size: 원본 이미지 크기 (width, height)
        
    Returns:
        numpy array: 원본 크기로 복원된 disparity
    """
    try:
        # 데이터 범위 정규화 (0-1)
        disparity_norm = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-8)
        
        # 원본 크기로 복원
        disparity_pil = pil.fromarray((disparity_norm * 255).astype(np.uint8))
        disparity_resized = disparity_pil.resize(original_size, pil.LANCZOS)
        
        return np.array(disparity_resized) / 255.0
        
    except Exception as e:
        print(f"⚠️ Disparity 후처리 오류: {e}")
        return disparity

def save_disparity_results(disparity, image_path, output_dir):
    """
    Disparity 결과 저장
    
    Args:
        disparity: disparity map
        image_path: 원본 이미지 경로
        output_dir: 출력 디렉토리
    """
    try:
        # 출력 파일명 생성
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. NumPy 파일 저장
        npy_path = os.path.join(output_dir, f"{output_name}_disp.npy")
        np.save(npy_path, disparity)
        
        # 2. 컬러맵 이미지 저장
        vmax = np.percentile(disparity, 95)
        vmin = np.percentile(disparity, 5)
        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disparity)[:, :, :3] * 255).astype(np.uint8)
        
        im = pil.fromarray(colormapped_im)
        img_path = os.path.join(output_dir, f"{output_name}_disp.jpeg")
        im.save(img_path)
        
        return npy_path, img_path
        
    except Exception as e:
        print(f"⚠️ 결과 저장 오류: {e}")
        return None, None

def test_tensorrt_depth(args):
    """TensorRT를 사용한 깊이 추정 메인 함수"""
    
    # TensorRT 엔진 로드
    print(f"-> TensorRT 엔진 로드: {args.engine_path}")
    
    try:
        depth_estimator = TensorRTDepthEstimatorFixed(args.engine_path)
    except Exception as e:
        print(f"❌ TensorRT 엔진 로드 실패: {e}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 이미지 경로 처리
    image_paths = []
    
    if isinstance(args.image_path, list):
        for path in args.image_path:
            if os.path.isdir(path):
                image_paths.extend(glob.glob(os.path.join(path, f'*.{args.ext}')))
            elif os.path.isfile(path):
                image_paths.append(path)
    else:
        if os.path.isdir(args.image_path):
            image_paths = glob.glob(os.path.join(args.image_path, f'*.{args.ext}'))
        elif os.path.isfile(args.image_path):
            image_paths = [args.image_path]
        else:
            print(f"⚠️ 이미지 경로를 찾을 수 없습니다: {args.image_path}")
            return
    
    # disparity 이미지 제외
    image_paths = [p for p in image_paths if not (p.endswith("_disp.jpg") or p.endswith("_disp.jpeg"))]
    
    print(f"-> 처리할 이미지: {len(image_paths)}개")
    
    if not image_paths:
        print("처리할 이미지가 없습니다.")
        return
    
    # 배치 처리
    batch_size = args.batch_size
    total_images = len(image_paths)
    total_time = 0
    processed_count = 0
    
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_paths = image_paths[batch_start:batch_end]
        
        # 배치 데이터 준비
        batch_data = []
        original_sizes = []
        valid_paths = []
        
        for image_path in batch_paths:
            try:
                # 이미지 전처리
                image_tensor, original_size = preprocess_image(image_path)
                batch_data.append(image_tensor)
                original_sizes.append(original_size)
                valid_paths.append(image_path)
                
            except Exception as e:
                print(f"⚠️ 이미지 처리 실패 {os.path.basename(image_path)}: {e}")
                continue
        
        if not batch_data:
            continue
        
        # 배치 텐서 생성
        batch_tensor = np.stack(batch_data, axis=0)
        current_batch_size = batch_tensor.shape[0]
        
        # TensorRT 추론
        start_time = time.time()
        try:
            disparity_batch = depth_estimator.predict(batch_tensor)
            inference_time = time.time() - start_time
            total_time += inference_time
            
        except Exception as e:
            print(f"❌ TensorRT 추론 실패: {e}")
            continue
        
        # 결과 처리 및 저장
        for i in range(current_batch_size):
            image_path = valid_paths[i]
            original_size = original_sizes[i]
            disparity = disparity_batch[i, 0]  # (H, W)
            
            # 원본 크기로 복원
            disparity_resized = postprocess_disparity(disparity, original_size)
            
            # 결과 저장
            npy_path, img_path = save_disparity_results(disparity_resized, image_path, args.output_dir)
            
            processed_count += 1
            
            if img_path:
                print(f"   ✅ 처리 완료 ({processed_count}/{total_images}): {os.path.basename(image_path)}")
                print(f"      -> {os.path.basename(img_path)}")
        
        # 성능 정보 출력
        fps = current_batch_size / inference_time
        print(f"   배치 추론: {inference_time * 1000:.1f}ms (배치: {current_batch_size}), FPS: {fps:.1f}")
    
    # 전체 성능 통계
    if processed_count > 0:
        avg_time_per_image = (total_time / processed_count) * 1000  # ms
        total_fps = processed_count / total_time if total_time > 0 else 0
        
        print(f"\n=== ✅ 처리 완료 ===")
        print(f"처리된 이미지: {processed_count}/{total_images}")
        print(f"총 소요 시간: {total_time:.2f}초")
        print(f"평균 처리 시간: {avg_time_per_image:.1f}ms/이미지")
        print(f"전체 FPS: {total_fps:.1f}")
    else:
        print("❌ 처리된 이미지가 없습니다.")

def direct():
    """기본 이미지 디렉토리 반환"""
    directory = r"C:\Users\wodud\OneDrive\Desktop\도로주행 데이터\test"
    
    if os.path.exists(directory):
        folders = [os.path.join(directory, f) for f in os.listdir(directory) 
                  if os.path.isdir(os.path.join(directory, f))]
        
        if folders:
            print("디렉토리 내 폴더 목록:")
            for folder in folders:
                print(folder)
            return folders[0]  # 첫 번째 폴더 선택
        else:
            return directory
    else:
        return "./test_images"  # 기본값

def main():
    """메인 함수"""
    args = parse_args()
    
    try:
        test_tensorrt_depth(args)
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()